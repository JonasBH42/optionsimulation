import itertools
import json
import math
import numpy as np
import pytest

from qqq_sim.risk.contracts import RiskError,stable_hash
from qqq_sim.risk.pricing_fast import PriceBasis,PriceDecisionTree,PRICE_TREE_SCHEMA
from qqq_sim.risk.offline.price_basis import build_price_tree
from qqq_sim.risk.pack_loader import ModelPack,file_hash


def cells():
    axes=([-.2,0.,.2],[.1,.15,.2],[0.,.005,.01],[0.,.005,.01])
    bounds=np.array(list(itertools.product(*(list(zip(axis[:-1],axis[1:])) for axis in axes))),dtype=np.float64)
    # A hole and nonspatial original indices test abstention and face priority.
    return bounds[np.random.default_rng(717).permutation(len(bounds))][1:]


def basis_arrays(bounds):
    coefficients=np.zeros((len(bounds),2,4,4,4,4),dtype=np.float64)
    coefficients[:,:,0,0,0,0]=.5
    return {'price_bounds':bounds,'price_coefficients':coefficients,
            'price_errors':np.zeros((len(bounds),2)),
            'price_greek_errors':np.zeros((len(bounds),2,4))}


def basis(arrays,tree=True):
    return PriceBasis(arrays['price_bounds'],arrays['price_coefficients'],arrays['price_errors'],arrays['price_greek_errors'],
        **({'tree_nodes':arrays['price_tree_nodes'],'tree_bounds':arrays['price_tree_bounds']} if tree else {}))


def test_saved_tree_exactly_matches_scan_on_interiors_holes_faces_and_nextafter():
    bounds=cells(); arrays={**basis_arrays(bounds),**build_price_tree(bounds)}
    scan,tree=basis(arrays,False),basis(arrays)
    rng=np.random.default_rng(81)
    points=list(rng.uniform([-.25,.08,-.002,-.002],[.25,.22,.012,.012],size=(300,4)))
    for cell in bounds:
        points.extend(itertools.product(*cell))
        center=np.mean(cell,axis=1)
        for axis in range(4):
            for endpoint in cell[axis]:
                point=center.copy(); point[axis]=endpoint; points.append(point.copy())
                point[axis]=np.nextafter(endpoint,-math.inf);points.append(point.copy())
                point[axis]=np.nextafter(endpoint,math.inf);points.append(point.copy())
    for point in points:
        assert tree.locate(point).tolist()==scan.locate(point).tolist()
    shared=np.array([0.,.15,.005,.005])
    assert tree.locate(shared).tolist()==list(range(len(bounds)))
    for right in ('CALL','PUT'):
        actual=tree.evaluate(right,1.,1.,1.,.005,.005,.15,greeks=True)
        assert actual==scan.evaluate(right,1.,1.,1.,.005,.005,.15,greeks=True)
        assert actual['cell']==0 and actual['shared_face_cells']==list(range(len(bounds)))
    repeated=build_price_tree(bounds)
    assert all(np.array_equal(arrays[name],value) for name,value in repeated.items())


def test_tree_retains_coarse_fine_face_candidates_and_price_jump_gate():
    bounds=np.array([[[-.2,0],[.1,.2],[0,.01],[0,.01]],
                     [[0,.2],[.1,.15],[0,.01],[0,.01]],
                     [[0,.2],[.15,.2],[0,.01],[0,.01]]],dtype=np.float64)
    arrays={**basis_arrays(bounds),**build_price_tree(bounds)}
    assert basis(arrays).locate([0,.15,.005,.005]).tolist()==[0,1,2]
    arrays['price_coefficients']=arrays['price_coefficients'].copy()
    arrays['price_coefficients'][1,0,0,0,0,0]=.6
    with pytest.raises(RiskError,match='PRICE_BASIS_CROSS_FACE_FAILURE'):
        basis(arrays).evaluate('CALL',1.,1.,1.,.005,.005,.15)


def test_online_lookup_prunes_large_tree_and_never_calls_full_scan(monkeypatch):
    axes=(np.linspace(-.2,.2,17),np.linspace(.1,.2,9),np.linspace(0,.01,5),np.linspace(0,.01,5))
    bounds=np.array(list(itertools.product(*(list(zip(axis[:-1],axis[1:])) for axis in axes))))
    saved=build_price_tree(bounds)
    tree=PriceDecisionTree(bounds,saved['price_tree_nodes'],saved['price_tree_bounds'])
    actual=tree.bounds
    class CountedBounds:
        reads=0
        def __getitem__(self,index):
            self.reads+=1; return actual[index]
    counted=CountedBounds();tree.bounds=counted
    def no_scan(*args,**kwargs): raise AssertionError('All-cell scan called online')
    monkeypatch.setattr(np,'flatnonzero',no_scan)
    assert len(tree.locate([.003,.133,.0033,.0066]))==1
    assert counted.reads<80 and counted.reads<len(bounds)//10
    import qqq_sim.risk.pricing_fast as module
    monkeypatch.setattr(module,'MAX_PRICE_LOOKUP_NODES',1)
    with pytest.raises(RiskError,match='PRICE_TREE_LOOKUP_LIMIT'):
        tree.locate([.003,.133,.0033,.0066])


@pytest.mark.parametrize('corruption',['cycle','duplicate_child','duplicate_cell','leaf_bounds','branch_bounds','float_indices','big_endian','nan','shape'])
def test_corrupt_saved_tree_is_rejected_before_query(corruption):
    bounds=cells();saved=build_price_tree(bounds)
    nodes=saved['price_tree_nodes'].copy(); boxes=saved['price_tree_bounds'].copy()
    leaf=np.flatnonzero(nodes[:,2]>=0)
    if corruption=='cycle': nodes[0,0]=0
    elif corruption=='duplicate_child': nodes[0,1]=nodes[0,0]
    elif corruption=='duplicate_cell': nodes[leaf[0],2]=nodes[leaf[1],2]
    elif corruption=='leaf_bounds': boxes[leaf[0],0,0]-=.001
    elif corruption=='branch_bounds': boxes[0,0,0]-=.001
    elif corruption=='float_indices': nodes=nodes.astype(np.float64)
    elif corruption=='big_endian': nodes=nodes.astype('>i8')
    elif corruption=='nan': boxes[0,0,0]=np.nan
    elif corruption=='shape': nodes=nodes[:-1]
    with pytest.raises(RiskError,match='PRICE_TREE_INVALID'):
        PriceDecisionTree(bounds,nodes,boxes)


def write_pack(folder,arrays,qualified=False,tree_schema=PRICE_TREE_SCHEMA):
    folder.mkdir(exist_ok=True); artifacts={}
    for name,value in arrays.items():
        path=folder/(name+'.npy');np.save(path,value,allow_pickle=False)
        artifacts[name]={'file':path.name,'bytes':path.stat().st_size,'sha256':file_hash(path)}
    manifest={'schema_version':'QQQ_P50_PACK_V1','pack_id':'STRUCTURAL_TREE_TEST_ONLY','artifacts':artifacts}
    if any(name.startswith('price_tree_') for name in arrays): manifest['price_tree_schema']=tree_schema
    # Controlled gate flag tests structural refusal; this is no numerical
    # qualification fixture and is never published as a production pack.
    validation={'manifest_hash':stable_hash(manifest),'capabilities':{'price':{'qualified':qualified}}}
    for stem,value in (('manifest',manifest),('validation',validation)):
        path=folder/(stem+'.json');path.write_text(json.dumps(value),encoding='utf-8')
        (folder/(stem+'.sha256')).write_text(file_hash(path),encoding='ascii')
    return folder


def test_qualified_price_requires_saved_tree_but_legacy_unqualified_fixture_loads(tmp_path):
    arrays=basis_arrays(cells())
    legacy=ModelPack.load(write_pack(tmp_path/'legacy',arrays))
    assert legacy.price_basis.tree is None
    with pytest.raises(RiskError,match='PRICE_DECISION_TREE_MISSING'):
        ModelPack.load(write_pack(tmp_path/'missing',arrays,qualified=True))
    arrays.update(build_price_tree(arrays['price_bounds']))
    loaded=ModelPack.load(write_pack(tmp_path/'tree',arrays,qualified=True))
    loaded.require_qualified('price')
    assert loaded.price_basis.tree is not None
    assert loaded.price_basis.locate([0,.15,.005,.005]).tolist()==list(range(len(arrays['price_bounds'])))


def test_tree_integrity_receipts_do_not_replace_structural_validation(tmp_path):
    arrays=basis_arrays(cells());arrays.update(build_price_tree(arrays['price_bounds']))
    path=write_pack(tmp_path/'tree',arrays)
    raw=bytearray((path/'price_tree_nodes.npy').read_bytes());raw[-1]^=1
    (path/'price_tree_nodes.npy').write_bytes(raw)
    with pytest.raises(RiskError,match='PACK_INTEGRITY_FAILURE'):
        ModelPack.load(path)
    arrays['price_tree_nodes']=arrays['price_tree_nodes'].copy();arrays['price_tree_nodes'][0,0]=0
    path=write_pack(path,arrays)
    with pytest.raises(RiskError,match='PRICE_TREE_INVALID'):
        ModelPack.load(path)


def test_tree_schema_and_partial_artifacts_are_not_silently_ignored(tmp_path):
    arrays=basis_arrays(cells());arrays.update(build_price_tree(arrays['price_bounds']))
    with pytest.raises(RiskError,match='PRICE_TREE_INVALID'):
        ModelPack.load(write_pack(tmp_path/'wrong-schema',arrays,tree_schema='UNKNOWN'))
    arrays.pop('price_tree_bounds')
    with pytest.raises(RiskError,match='PRICE_TREE_INVALID'):
        ModelPack.load(write_pack(tmp_path/'partial',arrays))


def test_saved_price_values_are_not_silently_reinterpreted_as_float64(tmp_path):
    arrays=basis_arrays(cells());arrays.update(build_price_tree(arrays['price_bounds']))
    arrays['price_coefficients']=arrays['price_coefficients'].astype(np.float32)
    with pytest.raises(RiskError,match='INVALID_PRICE_BASIS'):
        ModelPack.load(write_pack(tmp_path/'wrong-price-dtype',arrays,qualified=True))
