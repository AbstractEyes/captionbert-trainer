"""Build concise frozen-control tables from the completed pinned measurements."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(path):return json.loads(Path(path).read_text(encoding="utf8"))
def table(headers,rows):return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]+["| "+" | ".join(map(str,row))+" |" for row in rows])
def f(v,d=5):return f"{v:.{d}f}"
def main():
 pairs=[]
 for seed in (1,2):
  folder=ROOT/f".cache/structured-results/KALLP-seed{seed}"
  j=load(folder/"frozen-followup-59ec95344958/report.json")
  phase=load(folder/"frozen-phase-constant-20261106/report.json")
  assert all(j[k]==phase[k] for k in ("arm","seed","config_id","dataset_id","row_ids_sha256"))
  result=load(folder/"result.json")
  assert phase["checkpoint_sha256"]==result["checkpoint_sha256"]
  assert j["independent_checkpoint_auto_parity"]["state_tensors_equal"]
  assert j["measurements"]["intact"]==phase["measurements"]["intact"]
  j["measurements"].update(phase_constant=phase["measurements"]["phase_constant"])
  j["sentence_measurements"].update(phase_constant=phase["sentence_measurements"]["phase_constant"])
  j["isolated_phase_source_sha256"]=phase["probe_source_sha256"]
  pairs.append(j)
 labels={"intact":"Intact trained head","phase_constant":"All lexical phases identical; norm retained","binding_off":"Binding path off; also bypasses norm","norm_off":"Record norm off; phases retained","phase_redraw":"New fixed phase draw; key equality retained","initialization":"Reconstructed fresh initialization"}
 rows=[];sentence=[]
 for condition in labels:
  values=[j["initialization_control"]["measurements"]["all"] if condition=="initialization" else j["measurements"][condition]["all"] for j in pairs]
  rows.append([labels[condition],*[f(x["exact_record_top1"]*100,3) for x in values],
               *[f(x["record_mass"],6) for x in values],
               *[f(x["caption_mean_record_mass"],6) for x in values]])
  if condition!="initialization":
   sentence.append([labels[condition],*[f(j["sentence_measurements"][condition]["development"]["spearman"]) for j in pairs],
      *[f(j["sentence_measurements"][condition]["evaluation"]["heldout"]["cos_to_target"],6) for j in pairs]])
 tables={"FROZEN_LOOKUP_TABLE":table(["Condition","Top-one seed 1 (%)","Top-one seed 2 (%)","Query mean probability 1","Query mean probability 2","Caption mean probability 1","Caption mean probability 2"],rows),
 "FROZEN_SENTENCE_TABLE":table(["Condition","Dev STS-B seed 1","Dev STS-B seed 2","Teacher cosine seed 1","Teacher cosine seed 2"],sentence)}
 hierarchy=[];records={}
 for stage in ("full4m","exposure1m"):
  for seed in (1,2):
   key=f"{stage}/H16-seed{seed}"
   j=load(ROOT/f".cache/structured-final-results/{key}/frozen-followup-59ec95344958/report.json")
   records[key]=j
   intact=j["sentence_measurements"]["intact"]["development"]["spearman"]
   for condition, label in (("intact","Intact"),("anchor_angles_off","Address rotations off"),("value_addresses_constant","Value address variation removed")):
    x=j["sentence_measurements"][condition]
    hierarchy.append(["4M ×4" if stage=="full4m" else "1M ×16",seed,label,
     f(x["development"]["spearman"]),f(x["development"]["spearman"]-intact),
     f(x["evaluation"]["heldout"]["cos_to_target"],6),f(x["evaluation"]["heldout"]["erank"],2)])
 tables["HIERARCHY_FROZEN_TABLE"]=table(["Recipe","Seed","Condition","Dev STS-B","Change from intact","Teacher cosine","Participation rank"],hierarchy)
 (ROOT/".cache/article-tables/frozen-tables.json").write_text(json.dumps(tables,indent=2)+"\n",encoding="utf8",newline="\n")
 summary={"binding":pairs,"hierarchical":records,"source_artifact_revision":load(ROOT/"docs/results/structured_artifact_census.json")["source_revision"],
  "frozen_development_only":True,"new_training_performed":False}
 (ROOT/"docs/results/structured_frozen_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf8",newline="\n")
 print("Built frozen lookup, sentence, hierarchy tables from actual reports.")
if __name__=="__main__":main()
