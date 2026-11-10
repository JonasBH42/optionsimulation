"""Index every specification block and acceptance ID without inventing passes."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
ROOT=Path(r"D:\optionsimulation")
sys.path.insert(0,str(ROOT/"src"))
from qqq_sim.serialization import save_json

ID_PATTERN=r"\b(?:SC|APP|DT|AT|PT|OT|DATA|DEC|EX|GAP|ID|INPUT|PR|RES)-\d{2}\b|\b(?:P50-[RM]|APP-C)\d{2}\b"
EVIDENCE_STATUS="SCOPED_EVIDENCE_MAPPED; REQUIREMENT_ACCEPTANCE_NOT_ASSERTED"


def main():
    text=(ROOT/"qqq_simulator_5_part_implementation_plan_complete.txt").read_text(encoding="utf-8")
    records={}
    block="PREAMBLE"
    sections=[]
    for number,line in enumerate(text.splitlines(),1):
        match=re.match(r"BEGIN (SRC-[A-E]|P50-[RM])",line)
        if match: block=match[1]
        if number>=6124 and re.match(r"\d\d\. ",line): block="APP"
        heading=re.match(r"(\d\d)\. (.+)$",line)
        if heading and heading[2].isupper():
            sections.append({"id":f"{block}/{heading[1]}","line":number,"title":heading[2],"status":"SEE_REQUIREMENT_AND_DOMAIN_EVIDENCE"})
        subsection=re.match(r"(\d{1,2}\.\d+(?:\.\d+)?)[. ]\s*(\S.*)",line)
        if subsection and subsection[1].split('.')[-1]!='00':
            sections.append({"id":f"{block}/{subsection[1]}","line":number,"title":subsection[2],"status":"SEE_REQUIREMENT_AND_DOMAIN_EVIDENCE"})
        local_id=re.match(r"((?:T|U)\d{1,2}|(?:B|Q)\d)(?:\.\s+|\s+-\s+|\s+(?=[A-Z]))",line)
        if local_id:
            identifier=f"{block}/{local_id[1]}"
            records.setdefault(identifier,{"id":identifier,"source_lines":[],"status":"NOT_INDEPENDENTLY_VERIFIED","evidence":[]})["source_lines"].append(number)
        for match in re.finditer(ID_PATTERN,line):
            identifier=match[0]
            record=records.setdefault(identifier,{"id":identifier,"source_lines":[],"status":"NOT_INDEPENDENTLY_VERIFIED","evidence":[]})
            record["source_lines"].append(number)
            if line.startswith(identifier) and re.match(r"[. ]",line[len(identifier):]):
                record.setdefault("definition_lines",[]).append(number)
                record.setdefault("definition",line)
    for file in (ROOT/"docs/data-traceability.json",ROOT/"docs/account_coverage.json",ROOT/"docs/risk_coverage.json"):
        if not file.exists(): continue
        coverage=json.loads(file.read_text(encoding="utf-8"))
        raw=json.dumps(coverage)
        for identifier,record in records.items():
            if re.search(r"(?<![A-Za-z0-9])"+re.escape(identifier)+r"(?![A-Za-z0-9])",raw):
                record["evidence"].append(file.relative_to(ROOT).as_posix())
                record["status"]=EVIDENCE_STATUS
    test_map={
        "APP-01":["tests/test_ui.py","src/qqq_sim/ui/chart.py"],
        "APP-02":["tests/test_replay_integration.py","src/qqq_sim/ui/main_window.py"],
        "APP-03":["tests/test_data.py","src/qqq_sim/data/underlying.py"],
        "APP-04":["tests/test_ui.py","src/qqq_sim/ui/chain.py"],
        "APP-05":["tests/test_account_broker.py","tests/test_replay_integration.py"],
        "APP-06":["tests/test_account_analysis_inputs.py","src/qqq_sim/services/analysis_inputs.py"],
        "APP-07":["src/qqq_sim/risk/envelope.py","src/qqq_sim/ui/worker.py"],
        "APP-08":["tests/test_account_broker.py","src/qqq_sim/accounting/ledger.py"],
        "APP-09":["tests/test_ui.py","tests/test_replay_integration.py"],
        "APP-10":["tests/test_lifecycle_integration.py","tests/test_replay_integration.py"],
        "APP-11":["tests/test_storage_events.py","tests/test_replay_integration.py"],
        "APP-12":["tests/test_data.py","src/qqq_sim/ui/worker.py"],
        "APP-13":["tests/test_replay_integration.py"],
        "APP-14":["src/qqq_sim/config.py","tests/test_risk_policy.py"],
        "APP-15":["README.md","requirements.lock","runtime/results/release-manifest.json"]}
    for identifier,files in test_map.items():
        if identifier in records:
            records[identifier]["evidence"]+=files
            records[identifier]["status"]=EVIDENCE_STATUS
    coverage=json.loads((ROOT/'docs/application-data-coverage.json').read_text(encoding='utf-8'))
    for group in coverage['mappings']:
        identifiers=group.get('requirements',[])+[key for key in records if any(key.startswith(prefix) for prefix in group.get('prefixes',[]))]
        for identifier in identifiers:
            if identifier in records:
                records[identifier]['evidence']+=group['evidence']
                records[identifier]['scope_notes']=group['scope']
                records[identifier]['status']=EVIDENCE_STATUS
    for identifier,record in records.items():
        if identifier.startswith("P50-M") or identifier in ("PT-07","PT-08","PT-13","PT-14","PT-16","PT-22","PT-23","PT-24","PT-27","PT-29","PT-30","SC-01","SC-02","SC-03","SC-06","SC-07","SC-09","SC-13","APP-07"):
            record["acceptance_gate"]="PRODUCTION_NUMERICAL_QUALIFICATION_OUTSTANDING; SEE_SCOPED_RISK_REPORT"
            record["evidence"].append("runtime/results/validation-fixture-report.json")
        if identifier in ("OT-09","OT-11","OT-12","APP-15","APP-C13"):
            record["acceptance_gate"]="CLEAN_ACCOUNT_OR_SUPPORTED_QUERY_RELEASE_GATES_OUTSTANDING; SEE_RELEASE_STATUS"
        record["source_lines"]=sorted(set(record["source_lines"]))
        record["evidence"]=sorted(set(record["evidence"]))
        record['missing_evidence_paths']=[path for path in record['evidence'] if not (ROOT/path).exists()]
        record['record_kind']='RUNTIME_INPUT_OR_QUALIFICATION_DECLARATION' if identifier.startswith(('GAP-','INPUT-')) else 'PRECEDENCE_DECISION_OR_IDENTITY_CONTRACT' if identifier.startswith(('PR-','RES-','DEC-','ID-')) else 'REQUIREMENT_OR_ACCEPTANCE_CASE'
    result={"schema_version":2,"specification":"qqq_simulator_5_part_implementation_plan_complete.txt",
            "interpretation":"This is an explicit-ID and numbered-section navigation index, not an exhaustive decomposition of every prose obligation or an acceptance certificate. Evidence links identify scoped tests/implementations; the generator does not run tests or assert that every clause passed. Real archive metadata, production numerical qualification and clean-account release gates remain separate.",
            "release_gate_source":"docs/release-status.json",
            "status_semantics":{EVIDENCE_STATUS:"A relevant implementation, test, or measured report is linked; no whole-requirement completion is inferred.","NOT_INDEPENDENTLY_VERIFIED":"No scoped evidence mapping is recorded; this is neither a pass nor proof code is absent."},
            "mapping_audit":{"application_and_data_ids_without_evidence":[identifier for identifier,record in records.items() if identifier.startswith(('APP-','DATA-','DT-')) and not record['evidence']],"missing_evidence_paths":sorted({path for record in records.values() for path in record['missing_evidence_paths']})},
            "sections":sections,"requirements":list(sorted(records.values(),key=lambda r:r["id"])),
            "conditional_adapters":["Authenticated read-only broker transport","SPX European product transport/GTH adapter","Android/Termux compatibility","Exact account-specific broker parity"],
            "milestones":[{"id":f"M{i}","source_section":f"APP/28.{i+1}","status":"SCOPED_IMPLEMENTATION_EVIDENCE; MILESTONE_GATE_NOT_CERTIFIED" if i<5 else "NUMERICAL_QUALIFICATION_INCOMPLETE" if i<8 else "SUPPORTED_WORKFLOW_AND_RELEASE_GATES_OUTSTANDING","evidence":["docs/data-traceability.json","docs/account_coverage.json","docs/risk_coverage.json","docs/release-status.json"]} for i in range(10)]}
    save_json("docs/requirements.json",result)
    print(json.dumps({"indexed_requirements":len(records),"indexed_sections":len(sections)}))


if __name__=="__main__": main()
