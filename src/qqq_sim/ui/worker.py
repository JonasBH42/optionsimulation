"""Qt's serialized simulation writer. No widgets or models are touched here."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot

from qqq_sim.cli import error_record
from qqq_sim.serialization import digest, primitive
from qqq_sim.services.analysis import AnalysisIdentity


class SimulationWorker(QObject):
    completed = Signal(str, object)
    failed = Signal(str, object)
    advance_progress = Signal(object)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.controller = None

    @Slot(object)
    def execute(self, command):
        kind = command["kind"]
        try:
            if kind in ("open", "restore", "demo"):
                from qqq_sim.replay.controller import ReplayController
                config = self.config
                if kind == "demo":
                    from qqq_sim.cli import demo_config
                    config = demo_config(config)
                new = (ReplayController.restore(config,command["path"]) if kind == "restore"
                       else ReplayController.create(config,command.get("at")))
                try:
                    result=new.bundle()
                    if self.controller:
                        self.controller.close()
                except Exception:
                    new.store.connection.close()
                    if self.controller:
                        try:
                            self.controller.store.connection.execute('SELECT 1')
                        except Exception:
                            self.controller=ReplayController.restore(self.controller.config,self.controller.store.path,self.controller.branch_id)
                    raise
                self.controller = new
                self.config=new.config
            elif kind == "advance":
                result = self.controller.advance(target=command.get("target"),minutes=command.get("minutes",1),
                                                 expected_version=command["version"],command_id=command["id"],
                                                 progress=self.advance_progress.emit)
            elif kind in ("submit", "preview"):
                result = self.controller.submit(command["draft"],command["version"],command["id"],preview=kind=="preview")
            elif kind == "cancel":
                result = self.controller.cancel(command["order_id"],command["version"],command["id"])
            elif kind == "stock_reduce":
                result=self.controller.submit_stock_reduction(command["campaign_id"],command["quantity"],command["minimum_cash_per_share"],command["version"],command["id"])
            elif kind == 'prepare_benchmark':
                result=self.controller.prepare_benchmark(command['campaign_id'])
            elif kind == 'record_benchmark':
                result=self.controller.record_benchmark(command['campaign_id'],command['benchmark_id'],command['result'],command['version'],command['id'])
            elif kind == "branch":
                result = self.controller.branch(command.get("sequence"),command.get("reason","User inspected future before branching"))
                self.config=self.controller.config
            elif kind == "export":
                result = {"path":str(self.controller.export(command["path"]))}
            elif kind == "record_analysis":
                if self.controller is not None:
                    result=self.controller.record_analysis(command['identity'],command['result'],command.get('request_hash'),command.get('archive_path'))
                else:
                    from qqq_sim.services.analytic_records import record_archived_analysis
                    result=record_archived_analysis(command['archive_path'],command['identity'],command['result'],command.get('request_hash'))
            elif kind == "analyze":
                c = self.controller
                c._expect(command["version"])
                config = c.config
                if not config.pack_path:
                    from qqq_sim.services.analytic_records import archive_unavailable
                    result=archive_unavailable(c,command['draft'],command['selection_hash'],
                        {"status":"UNSUPPORTED","reason":"PACK_MISSING","hold_probability":None,"managed_probability":None},
                        current_position=command.get('current_position',False))
                else:
                    if c._read_pack_hash()!=c._pack_hash:
                        from qqq_sim.broker.models import BrokerError
                        raise BrokerError("PACK_REVISION_CHANGED")
                    from qqq_sim.services.analysis_inputs import prepare_analysis
                    result=prepare_analysis(c,command["draft"],command["selection_hash"],current_position=command.get('current_position',False))

            elif kind == "load_pack":
                from qqq_sim.risk.pack_loader import ModelPack
                pack=ModelPack.load(command["path"])
                if self.controller:
                    result=self.controller.select_inputs(pack_path=command["path"])
                    self.config=self.controller.config
                else:
                    self.config=replace(self.config,pack_path=command["path"])
                    result={"status":"OK","pack_path":command["path"],"manifest":getattr(pack,"manifest",{})}
            elif kind == "load_parameters":
                json.loads(Path(command["path"]).read_text(encoding="utf-8-sig"))
                if self.controller:
                    result=self.controller.select_inputs(parameters_path=command["path"])
                    self.config=self.controller.config
                else:
                    self.config=replace(self.config,parameters_path=command["path"])
                    result={"status":"OK","parameters_path":command["path"]}
            elif kind == "close":
                if self.controller:
                    self.controller.close()
                    self.controller=None
                result={"status":"OK"}
            else:
                raise ValueError(f"Unknown worker command {kind}")
            if kind=='analyze' and isinstance(result,dict):
                result.setdefault('selection_hash',command.get('selection_hash'))
                result.setdefault('state_version',command.get('version'))
            self.completed.emit(kind,result)
        except Exception as exc:
            error=error_record(exc)
            if kind=='analyze':
                if hasattr(exc,'analysis_record'):
                    error.update(exc.analysis_record)
                elif self.controller is not None:
                    try:
                        from qqq_sim.services.analytic_records import archive_unavailable
                        error=archive_unavailable(self.controller,command.get('draft'),command.get('selection_hash'),error,
                            current_position=command.get('current_position',False),request_context={'requested_state_version':command.get('version')})
                    except Exception as archive_error:
                        error['archive_error']=error_record(archive_error)
                error['selection_hash']=command.get('selection_hash')
                error['state_version']=command.get('version')
            self.failed.emit(kind,error)
