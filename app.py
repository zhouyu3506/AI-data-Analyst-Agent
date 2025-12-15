import os
import json
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
from google.cloud import bigquery
from openai import OpenAI




DEFAULT_PROJECT = "MY_PROJECT"
DEFAULT_DATASET = "MY_DATASET"
DEFAULT_TABLE = "MY_TABLE"

# pipeline: Gemini -> SQL -> BigQuery -> DataFrame -> 图表 -> Gemini 总结

# 把字段映射到你的表结构
# order_date: 日期字段（date/timestamp)
# sales: 销售额字段
DEFAULT_SCHEMA_HINT = {
    "table": f"{DEFAULT_PROJECT}.{DEFAULT_DATASET}.{DEFAULT_TABLE}",
    "date_field":"order_date",
    "sales_expr": "sales_amount"
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 1. transmit natural language to sql
def llm_generate_sql(user_intent:str, schema_hint:dict) -> dict:

        
    prompt =  f"""
 You are a senior analytics engineer.
 Generate a BigQuery Standard SQL query for this intent:
 
 INTENT:
 {user_intent}
 
 SCHEMA:
  {json.dumps(schema_hint, indent=2)}
  
Rules:
- Use DATE({schema_hint["date_field"]}) as date
- Output daily aggregated sales
- Limit to last 7 days
  
Return STRICT JSON ONLY:
  {{"sql":"...", "explanation":"..."}}
  """
    resp = client.chat.completions.create(
         model="gpt-4.1-mini",
         messages=[
             {"role": "system","content": "You are a precise data engineer"},
             {"role":"user","content":prompt},
         ],
         temperature=0
    )
    text = resp.choices[0].message.content.strip()
    
    start = text.find("{")
    end = text.rfind("}") + 1
    
    return json.loads(text[start:end])

# 3. transmit data to natural language 
def llm_summarized(df: pd.DataFrame) -> str:
    df_for_llm = df.copy()
    df_for_llm["date"] = df_for_llm["date"].astype(str)    
    prompt = f"""
    Summarize the following 7-day sales trend.
    Be concise and business-oriented.
    Data : 
    {json.dumps(df_for_llm.to_dict(orient = "records"), indent=2)}
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages = [{"role": "user", "content":prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# 2. exec sql, and get the data
def run_bigquery(sql: str) -> pd.DataFrame:
    client = bigquery.Client()
    job = client.query(sql)
    df = job.result().to_dataframe()
    return df 

# streamlit: UI-driven workflow manager
# 收集参数 → 调 Gemini 生成 SQL → 调 BigQuery 查数据 → 画图 → 再调 Gemini 写总结
st.set_page_config(page_title="7-Day Sales Trend (Gemini + BigQuery + Streamlit)", layout="wide")
st.write("DEPLOY VERSION: 2025-01-DEPLOY-TEST")
st.title("Sales trend over the last 7 days")

with st.sidebar:
    st.header("BigQuery Table Settings")
    project = st.text_input("Project", DEFAULT_PROJECT)
    dataset = st.text_input("Dataset", DEFAULT_DATASET)
    table = st.text_input("Table", DEFAULT_TABLE)
    date_field = st.text_input("Date field", DEFAULT_SCHEMA_HINT["date_field"])
    sales_expr = st.text_input("Sales expression/field", DEFAULT_SCHEMA_HINT["sales_expr"])
    
    st.divider()
    st.header("Intent")
    user_intent = st.text_area(
        "What do you want?",
        value = "Show sales trend over the last 7 days(daily total sales).",
        height =80
    )
    
schema_hint = {
    "table": f"{project}.{dataset}.{table}",
    "date_field": date_field,
    "sales_expr": sales_expr
}    

colA, colB = st.columns([1,1])

with colA:
    if st.button("1. LLM -> Generate SQL", use_container_width= True):
        try:
            out = llm_generate_sql(user_intent, schema_hint)
            st.session_state["generated_sql"] = out["sql"]
            st.session_state["sql_explanation"] = out.get("explanation", "")
            st.success("SQL generated.")
            
        except Exception as e:
            st.error(str(e))
            
with colB:
    if st.button("2. BigQuery -> Run SQL", use_container_width= True):
       sql = st.session_state.get("generated_sql", "")
       if not sql.strip():
           st.error("Generate SQL first")
       else: 
           try:
               df = run_bigquery(sql)
               st.session_state["df"] = df
               st.success(f"Query completed. Rows: {len(df)}")
           except Exception as e:
               st.error(str(e))                       


st.subheader("Generated SQL")
sql = st.session_state.get("generated_sql", "")
st.code(sql or "--Click 'LLM -> Generated SQL' --", language="sql")

explain = st.session_state.get("sql_explanation", "")
if explain:
    st.caption(explain)
    
df = st.session_state.get("df", None)
if df is not None and len(df) > 0:
    st.subheader("Result (daily)")
    st.dataframe(df, use_container_width=True)
    
    # Plot
    st.subheader("Trend chart")
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df = plot_df.sort_values("date")
    
    fig = plt.figure()
    plt.plot(plot_df["date"], plot_df.iloc[:,1])
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation = 45)
    plt.tight_layout() 
    st.pyplot(fig, clear_figure=True)
    
    if st.button("3 LLM -> Summarize result", use_container_width=True):
      try:
        summary = llm_summarized(df)   
        st.session_state["summary"] = summary
        
      except Exception as e:
        st.error(str(e))
       
summary = st.session_state.get("summary", "")
if summary:
    st.subheader("LLM Summary")
    st.write(summary)
    
else:
    st.info("Run the query, then click 'LLM -> Summarize result'.")   
       
        
            


