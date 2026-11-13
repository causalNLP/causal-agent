import warnings, os, pandas as pd
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv('.env')
os.environ["LLM_MODEL"]="deepseek/deepseek-v4.1-flash"
from cais.config import get_llm_client
from cais.components.dataset_analyzer import analyze_dataset
from cais.components.query_interpreter import _create_identify_prompt
from cais.models import LLMIVars

IV=set(range(76,113)); MATCH={22,23,24,26,27,28,30,31,32,34,35,37,38,39,40}; DID={122,128}
meta = pd.read_csv("experiments/data/causcibench/metadata_csv/synthetic_input.csv")
llm = get_llm_client()
res=[]
for rid in sorted(IV|MATCH|DID):
    r = meta[meta["id"]==rid].iloc[0]
    a = analyze_dataset(dataset_path=f"experiments/data/causcibench/synthetic_data/{r['dataset_name']}", llm_client=None)
    prompt = _create_identify_prompt("instrumental variable", query=str(r["query"]), description=str(r["data_description"]),
        columns=a["columns"], categories=a["column_categories"], treatment=str(r["treatment"]), outcome=str(r["outcome"]))
    named=0; best=None
    for _ in range(2):
        out = llm.with_structured_output(LLMIVars).invoke(prompt)
        if out and out.instrument_variable: named+=1; best=out.instrument_variable
    res.append((rid, named, best))
iv_named=sum(1 for rid,n,_ in res if rid in IV and n>0)
m_abst=sum(1 for rid,n,_ in res if rid in MATCH and n==0)
d_abst=sum(1 for rid,n,_ in res if rid in DID and n==0)
print(f"IV rows (must name):    {iv_named}/37 kept naming")
print(f"MATCH rows (must abstain): {m_abst}/15 now abstain")
print(f"DID rows (must abstain):   {d_abst}/2 now abstain")
print("\nstill-naming MATCH/DID rows:", [(rid, best) for rid, n, best in res if rid in (MATCH|DID) and n>0])
print("abstaining IV rows:", [rid for rid, n, _ in res if rid in IV and n==0])
