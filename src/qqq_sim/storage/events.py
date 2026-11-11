"""SQLite FULL/WAL event store with idempotent commands and hash-checked recovery."""
from __future__ import annotations
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
import uuid
from ..broker.models import BrokerError,canonical,fingerprint,encode

SCHEMA_VERSION=1


def _encoded_json(value):
    """Canonical JSON for values already normalized by encode()."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _encoded_fingerprint(value):
    return sha256(_encoded_json(value).encode('utf-8')).hexdigest()


def _writable(path):
    path=Path(path).expanduser().resolve()
    root=Path('D:/optionsimulation').resolve()
    try: path.relative_to(root)
    except ValueError: raise BrokerError('WRITE_OUTSIDE_PROJECT_FORBIDDEN',path=str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


SCHEMA='''
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,schema_version INTEGER NOT NULL,
 config_json TEXT NOT NULL,config_hash TEXT NOT NULL,initial_cash_cents INTEGER NOT NULL,
 active_branch TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS branches(id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES sessions(id),
 parent_id TEXT REFERENCES branches(id),fork_sequence INTEGER NOT NULL,fork_state_hash TEXT NOT NULL,
 config_hash TEXT NOT NULL,mode TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS commands(branch_id TEXT NOT NULL REFERENCES branches(id),command_id TEXT NOT NULL,
 expected_version INTEGER NOT NULL,payload_json TEXT NOT NULL,status TEXT NOT NULL,result_json TEXT NOT NULL,
 PRIMARY KEY(branch_id,command_id));
CREATE TABLE IF NOT EXISTS events(branch_id TEXT NOT NULL REFERENCES branches(id),sequence INTEGER NOT NULL,
 event_id TEXT NOT NULL UNIQUE,utc_ns INTEGER NOT NULL,phase TEXT NOT NULL,event_type TEXT NOT NULL,
 schema_version INTEGER NOT NULL,payload_json TEXT NOT NULL,previous_hash TEXT NOT NULL,event_hash TEXT NOT NULL,
 PRIMARY KEY(branch_id,sequence));
CREATE TABLE IF NOT EXISTS cash_postings(event_id TEXT NOT NULL REFERENCES events(event_id),posting_index INTEGER NOT NULL,
 campaign_id TEXT,bucket TEXT NOT NULL,category TEXT NOT NULL,amount_cents INTEGER NOT NULL,
 obligation_id TEXT,due_date TEXT,PRIMARY KEY(event_id,posting_index));
CREATE TABLE IF NOT EXISTS checkpoints(branch_id TEXT NOT NULL REFERENCES branches(id),sequence INTEGER NOT NULL,
 state_json_or_path TEXT NOT NULL,state_hash TEXT NOT NULL,dataset_hash TEXT,pack_hash TEXT,rules_hash TEXT,
 PRIMARY KEY(branch_id,sequence));
CREATE TABLE IF NOT EXISTS analyses(request_hash TEXT PRIMARY KEY,branch_id TEXT NOT NULL REFERENCES branches(id),
 state_version INTEGER NOT NULL,selection_hash TEXT NOT NULL,pack_hash TEXT,status TEXT NOT NULL,payload_json_or_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analysis_requests(request_hash TEXT PRIMARY KEY,branch_id TEXT NOT NULL REFERENCES branches(id),
 state_version INTEGER NOT NULL,selection_hash TEXT NOT NULL,pack_hash TEXT,identity_json TEXT NOT NULL,
 payload_hash TEXT NOT NULL,snapshot_payload_hash TEXT NOT NULL,payload_json TEXT NOT NULL,snapshot_json TEXT NOT NULL,
 created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS analysis_requests_branch ON analysis_requests(branch_id,state_version);
CREATE TABLE IF NOT EXISTS projections(name TEXT NOT NULL,branch_id TEXT NOT NULL REFERENCES branches(id),sequence INTEGER NOT NULL,
 schema_version INTEGER NOT NULL,payload_json TEXT NOT NULL,PRIMARY KEY(name,branch_id));
CREATE INDEX IF NOT EXISTS events_type ON events(branch_id,event_type);
'''


class EventStore:
    def __init__(self,path):
        self.path=_writable(path)
        self.connection=sqlite3.connect(str(self.path),isolation_level=None,timeout=30)
        self.connection.row_factory=sqlite3.Row
        try:
            # Inspect compatibility before changing an unknown database's journal.
            version=self.connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0,SCHEMA_VERSION): raise BrokerError('STORAGE_SCHEMA_UNSUPPORTED',schema_version=version)
            self.connection.execute('PRAGMA foreign_keys=ON')
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.executescript(SCHEMA)
            self.connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        except Exception:
            self.connection.close()
            raise

    def create_session(self,config,session_id=None,initial_state=None):
        session_id=session_id or str(uuid.uuid4()); branch_id=str(uuid.uuid4())
        now=datetime.now(timezone.utc).isoformat(); config_json=canonical(config); config_hash=fingerprint(config)
        initial_cash=config.get('initial_cash_cents',500000)
        if isinstance(config.get('account'),dict): initial_cash=int(float(config['account'].get('initial_cash',5000))*100)
        state=initial_state or {}; state_record={'version':0,'state':state}
        db=self.connection; db.execute('BEGIN IMMEDIATE')
        try:
            db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?)',(session_id,SCHEMA_VERSION,config_json,config_hash,initial_cash,branch_id,now))
            db.execute('INSERT INTO branches VALUES(?,?,?,?,?,?,?)',(branch_id,session_id,None,0,'',config_hash,config.get('mode','SIMULATION')))
            db.execute('INSERT INTO projections VALUES(?,?,?,?,?)',('state',branch_id,0,SCHEMA_VERSION,canonical(state_record)))
            db.execute('INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?)',(branch_id,0,canonical(state_record),fingerprint(state_record),None,None,None))
            db.execute('COMMIT')
        except Exception: db.execute('ROLLBACK'); raise
        return {'session_id':session_id,'branch_id':branch_id,'version':0}

    def sessions(self):
        return [dict(row) for row in self.connection.execute('SELECT * FROM sessions ORDER BY created_at')]

    def _projection(self,branch_id):
        row=self.connection.execute("SELECT * FROM projections WHERE name='state' AND branch_id=?",(branch_id,)).fetchone()
        if row is None: raise BrokerError('BRANCH_NOT_FOUND',branch_id=branch_id)
        data=json.loads(row['payload_json']); data['sequence']=row['sequence']
        return data

    def transact(self,branch_id,command_id,expected_version,payload,events,result,state):
        db=self.connection; db.execute('BEGIN IMMEDIATE')
        try:
            old=db.execute('SELECT result_json FROM commands WHERE branch_id=? AND command_id=?',(branch_id,command_id)).fetchone()
            if old:
                db.execute('COMMIT'); return json.loads(old['result_json'])
            current=self._projection(branch_id)
            if current['version']!=expected_version:
                raise BrokerError('STALE_STATE_VERSION',expected=expected_version,actual=current['version'])
            sequence=current['sequence']
            prior=db.execute('SELECT event_hash FROM events WHERE branch_id=? ORDER BY sequence DESC LIMIT 1',(branch_id,)).fetchone()
            if prior: previous_hash=prior['event_hash']
            else: previous_hash=db.execute('SELECT fork_state_hash FROM branches WHERE id=?',(branch_id,)).fetchone()[0]
            new_version=expected_version+1
            encoded_state=encode(state)
            state_record={'version':new_version,'state':encoded_state}
            state_json=_encoded_json(state_record)
            committed={'event_id':f'commit:{command_id}','event_type':'STATE_COMMITTED','phase':'COMMIT',
                       'payload':state_record}
            complete=list(events)+[committed]
            for event in complete:
                source_id=event.get('event_id') or fingerprint(event)
                event_id=branch_id+':'+source_id
                # Repeated source events do not append or repost.
                if db.execute('SELECT 1 FROM events WHERE event_id=?',(event_id,)).fetchone(): continue
                sequence+=1
                event_payload=state_record if event is committed else encode(event.get('payload',event))
                event_json=state_json if event is committed else _encoded_json(event_payload)
                utc_ns=event.get('utc_ns')
                if utc_ns is None:
                    at=event.get('at')
                    if at:
                        stamp=datetime.fromisoformat(at) if isinstance(at,str) else at
                        if stamp.tzinfo is None: raise BrokerError('NAIVE_EVENT_TIMESTAMP')
                        utc_ns=int(stamp.timestamp()*1_000_000_000)
                    else: utc_ns=0
                phase=event.get('phase','ECONOMIC'); kind=event.get('event_type','EVENT')
                hashed={'branch_id':branch_id,'sequence':sequence,'event_id':event_id,'utc_ns':utc_ns,
                        'phase':phase,'event_type':kind,'schema_version':SCHEMA_VERSION,'payload':event_payload,'previous_hash':previous_hash}
                event_hash=_encoded_fingerprint(hashed)
                db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',(branch_id,sequence,event_id,utc_ns,phase,kind,SCHEMA_VERSION,event_json,previous_hash,event_hash))
                for index,posting in enumerate(event.get('postings',())):
                    db.execute('INSERT INTO cash_postings VALUES(?,?,?,?,?,?,?,?)',(event_id,index,posting.get('campaign_id'),posting.get('bucket','economic'),
                               posting.get('category',kind),posting['amount_cents'],posting.get('obligation_id'),posting.get('due_date')))
                previous_hash=event_hash
            db.execute('INSERT OR REPLACE INTO projections VALUES(?,?,?,?,?)',('state',branch_id,sequence,SCHEMA_VERSION,state_json))
            response={'version':new_version,'sequence':sequence,'result':encode(result),'state_hash':_encoded_fingerprint(encoded_state)}
            db.execute('INSERT INTO commands VALUES(?,?,?,?,?,?)',(branch_id,command_id,expected_version,canonical(payload),'COMPLETED',canonical(response)))
            if sequence//1000>current['sequence']//1000:
                self._checkpoint_row(branch_id,sequence,state_record)
            db.execute('COMMIT')
            return response
        except Exception:
            db.execute('ROLLBACK'); raise

    def _checkpoint_row(self,branch_id,sequence,state_record,dataset_hash=None,pack_hash=None,rules_hash=None):
        self.connection.execute('INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?,?,?,?)',
                                (branch_id,sequence,canonical(state_record),fingerprint(state_record),dataset_hash,pack_hash,rules_hash))

    def checkpoint(self,branch_id,dataset_hash=None,pack_hash=None,rules_hash=None):
        data=self._projection(branch_id); sequence=data.pop('sequence')
        self._checkpoint_row(branch_id,sequence,data,dataset_hash,pack_hash,rules_hash)
        return {'sequence':sequence,'state_hash':fingerprint(data)}

    def _verify_event(self,row,previous):
        payload=json.loads(row['payload_json'])
        hashed={key:row[key] for key in ('branch_id','sequence','event_id','utc_ns','phase','event_type','schema_version','previous_hash')}
        hashed['payload']=payload
        if row['previous_hash']!=previous or fingerprint(hashed)!=row['event_hash']:
            raise BrokerError('EVENT_HASH_MISMATCH',sequence=row['sequence'])
        return payload

    def restore(self,branch_id,through_sequence=None):
        branch=self.connection.execute('SELECT * FROM branches WHERE id=?',(branch_id,)).fetchone()
        if branch is None: raise BrokerError('BRANCH_NOT_FOUND')
        if branch['parent_id']:
            parent=self.restore(branch['parent_id'],through_sequence=branch['fork_sequence'])
            if parent['event_hash']!=branch['fork_state_hash']:
                raise BrokerError('BRANCH_PREFIX_HASH_MISMATCH')
        where='branch_id=?'; args=[branch_id]
        if through_sequence is not None: where+=' AND sequence<=?'; args.append(through_sequence)
        checkpoints=self.connection.execute('SELECT * FROM checkpoints WHERE '+where+' ORDER BY sequence DESC',args).fetchall()
        valid=None
        for checkpoint in checkpoints:
            state_record=json.loads(checkpoint['state_json_or_path'])
            if fingerprint(state_record)==checkpoint['state_hash']:
                valid=(checkpoint['sequence'],state_record); break
        if valid is None: raise BrokerError('NO_VALID_CHECKPOINT')
        state_sequence,state_record=valid
        previous=branch['fork_state_hash']; sequence=branch['fork_sequence']
        # Validate the complete local hash chain, including before the checkpoint.
        for row in self.connection.execute('SELECT * FROM events WHERE '+where+' ORDER BY sequence',args):
            payload=self._verify_event(row,previous); previous=row['event_hash']; sequence=row['sequence']
            if row['sequence']==state_sequence and row['event_type']=='STATE_COMMITTED' and payload!=state_record:
                raise BrokerError('CHECKPOINT_EVENT_STATE_MISMATCH')
            if row['sequence']>state_sequence and row['event_type']=='STATE_COMMITTED': state_record=payload
        return {'state':state_record['state'],'version':state_record['version'],'sequence':sequence,
                'state_hash':fingerprint(state_record['state']),'event_hash':previous,'branch_id':branch_id,
                'session_id':branch['session_id']}

    def branch(self,parent_id,config,reason,fork_sequence=None):
        recovered=self.restore(parent_id,through_sequence=fork_sequence)
        self.checkpoint(parent_id)
        new_id=str(uuid.uuid4()); state_record={'state':recovered['state'],'version':recovered['version']}
        db=self.connection; db.execute('BEGIN IMMEDIATE')
        try:
            db.execute('INSERT INTO branches VALUES(?,?,?,?,?,?,?)',(new_id,recovered['session_id'],parent_id,recovered['sequence'],
                       recovered['event_hash'],fingerprint(config),str(reason)))
            db.execute('INSERT INTO projections VALUES(?,?,?,?,?)',('state',new_id,recovered['sequence'],SCHEMA_VERSION,canonical(state_record)))
            self._checkpoint_row(new_id,recovered['sequence'],state_record)
            db.execute('UPDATE sessions SET active_branch=? WHERE id=?',(new_id,recovered['session_id']))
            db.execute('COMMIT')
        except Exception: db.execute('ROLLBACK'); raise
        return {'session_id':recovered['session_id'],'branch_id':new_id,'version':recovered['version'],'state':recovered['state']}

    def record_analysis(self,request_hash,branch_id,state_version,selection_hash,pack_hash,status,payload):
        values=(request_hash,branch_id,state_version,selection_hash,pack_hash,status,canonical(payload))
        db=self.connection; db.execute('BEGIN IMMEDIATE')
        try:
            previous=db.execute('SELECT * FROM analyses WHERE request_hash=?',(request_hash,)).fetchone()
            if previous is not None and tuple(previous)!=values:
                raise BrokerError('ANALYSIS_RESULT_CONFLICT',request_hash=request_hash)
            if previous is None:
                db.execute('INSERT INTO analyses VALUES(?,?,?,?,?,?,?)',values)
            db.execute('COMMIT')
        except Exception:
            db.execute('ROLLBACK'); raise
        return self.analysis_result(request_hash)

    def register_analysis_request(self,request_hash,identity,payload,snapshot):
        from ..serialization import digest,primitive
        identity,payload,snapshot=map(primitive,(identity,payload,snapshot))
        if digest(identity)!=request_hash:
            raise BrokerError('ANALYSIS_REQUEST_IDENTITY_MISMATCH')
        values=(request_hash,identity['branch_id'],identity['state_version'],identity['selection_hash'],identity['pack_id'],
                canonical(identity),digest(payload),digest(snapshot),canonical(payload),canonical(snapshot))
        db=self.connection; db.execute('BEGIN IMMEDIATE')
        try:
            branch=db.execute('SELECT session_id FROM branches WHERE id=?',(identity['branch_id'],)).fetchone()
            if branch is None or branch[0]!=identity['session_id']:
                raise BrokerError('ANALYSIS_BRANCH_IDENTITY_MISMATCH')
            previous=db.execute('SELECT * FROM analysis_requests WHERE request_hash=?',(request_hash,)).fetchone()
            if previous is not None and tuple(previous)[:-1]!=values:
                raise BrokerError('ANALYSIS_REQUEST_CONFLICT',request_hash=request_hash)
            if previous is None:
                db.execute('INSERT INTO analysis_requests VALUES(?,?,?,?,?,?,?,?,?,?,?)',(*values,datetime.now(timezone.utc).isoformat()))
            db.execute('COMMIT')
        except Exception:
            db.execute('ROLLBACK'); raise
        return self.analysis_request(request_hash)

    def analysis_request(self,request_hash):
        from ..serialization import digest
        row=self.connection.execute('SELECT * FROM analysis_requests WHERE request_hash=?',(request_hash,)).fetchone()
        if row is None:
            raise BrokerError('ANALYSIS_REQUEST_NOT_FOUND',request_hash=request_hash)
        record=dict(row)
        identity=json.loads(record.pop('identity_json')); payload=json.loads(record.pop('payload_json')); snapshot=json.loads(record.pop('snapshot_json'))
        fields=('branch_id','state_version','selection_hash')
        if (digest(identity)!=request_hash or digest(payload)!=record['payload_hash'] or digest(snapshot)!=record['snapshot_payload_hash']
            or any(record[key]!=identity.get(key) for key in fields) or record['pack_hash']!=identity.get('pack_id')):
            raise BrokerError('ANALYSIS_REQUEST_HASH_MISMATCH',request_hash=request_hash)
        return {**record,'identity':identity,'payload':payload,'snapshot':snapshot}

    def analysis_result(self,request_hash):
        from ..serialization import digest
        row=self.connection.execute('SELECT * FROM analyses WHERE request_hash=?',(request_hash,)).fetchone()
        if row is None:
            return None
        record=dict(row); payload=json.loads(record.pop('payload_json_or_path'))
        if self.connection.execute('SELECT 1 FROM analysis_requests WHERE request_hash=?',(request_hash,)).fetchone():
            request=self.analysis_request(request_hash)
            if (payload.get('request_hash')!=request_hash or payload.get('result_hash')!=digest(payload.get('result'))
                or payload.get('identity')!=request['identity'] or payload.get('request_payload_hash')!=request['payload_hash']
                or payload.get('snapshot_payload_hash')!=request['snapshot_payload_hash']
                or any(record[key]!=request[key] for key in ('branch_id','state_version','selection_hash','pack_hash'))
                or record['status']!=str(payload.get('result',{}).get('status','UNAVAILABLE'))):
                raise BrokerError('ANALYSIS_RESULT_HASH_MISMATCH',request_hash=request_hash)
        return {**record,'payload':payload}

    def analysis_records(self,*,session_id=None,branch_id=None):
        if session_id is not None:
            rows=self.connection.execute('SELECT r.request_hash FROM analysis_requests r JOIN branches b ON b.id=r.branch_id WHERE b.session_id=? ORDER BY r.created_at,r.request_hash',(session_id,))
        elif branch_id is not None:
            rows=self.connection.execute('SELECT request_hash FROM analysis_requests WHERE branch_id=? ORDER BY created_at,request_hash',(branch_id,))
        else:
            rows=self.connection.execute('SELECT request_hash FROM analysis_requests ORDER BY created_at,request_hash')
        result=[]
        for row in rows:
            request=self.analysis_request(row[0]); observed=self.analysis_result(row[0])
            value=observed['payload']['result'] if observed else {}
            null_reason=value.get('reason') if observed else 'ANALYSIS_PENDING'
            result.append({**request,'status':observed['status'] if observed else 'PENDING','result_record':observed,
                           'null_reason':null_reason})
        return result

    def export(self,branch_id,destination=None):
        restored=self.restore(branch_id)
        branch=dict(self.connection.execute('SELECT * FROM branches WHERE id=?',(branch_id,)).fetchone())
        session=dict(self.connection.execute('SELECT * FROM sessions WHERE id=?',(branch['session_id'],)).fetchone())
        record={'schema_version':SCHEMA_VERSION,'session':session,'branch':branch,'restored':restored,
                'events':[dict(r) for r in self.connection.execute('SELECT * FROM events WHERE branch_id=? ORDER BY sequence',(branch_id,))],
                'commands':[dict(r) for r in self.connection.execute('SELECT * FROM commands WHERE branch_id=? ORDER BY rowid',(branch_id,))],
                'analyses':self.analysis_records(branch_id=branch_id)}
        record['export_hash']=fingerprint(record)
        if destination:
            path=_writable(destination); temp=path.with_suffix(path.suffix+'.tmp')
            with temp.open('w',encoding='utf-8',newline='\n') as handle:
                handle.write(canonical(record)); handle.flush()
                import os; os.fsync(handle.fileno())
            temp.replace(path)
        return record

    def backup(self,destination):
        target=sqlite3.connect(str(_writable(destination)))
        try: self.connection.backup(target)
        finally: target.close()

    def close(self):
        for row in self.connection.execute('SELECT id FROM branches').fetchall(): self.checkpoint(row[0])
        self.connection.close()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

