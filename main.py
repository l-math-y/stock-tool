import akshare as ak
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings("ignore")

app = FastAPI()
stock_list_cache = None

# 定义网页界面
html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>股票行情查询工具</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f0f2f5; color: #333; padding: 20px; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { text-align: center; margin-bottom: 20px; font-size: 24px; color: #1890ff; }
  .search-box { display: flex; gap: 10px; margin-bottom: 24px; background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
  .search-box input { flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; outline: none; }
  .search-box button { padding: 12px 30px; background: #1890ff; color: #fff; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; }
  .card { background: #fff; border-radius: 10px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
  .card-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #333; border-left: 4px solid #1890ff; padding-left: 10px; }
  .ret-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .ret-item { text-align: center; padding: 15px; background: #fafafa; border-radius: 8px; }
  .ret-label { font-size: 13px; color: #999; margin-bottom: 6px; }
  .ret-value { font-size: 20px; font-weight: 700; }
  .ret-value.up { color: #f5222d; }
  .ret-value.down { color: #52c41a; }
  .ret-value.neutral { color: #666; }
  .pred-box { text-align: center; padding: 20px; }
  .pred-dir { font-size: 32px; font-weight: 800; margin-bottom: 8px; }
  .pred-conf { font-size: 14px; color: #666; margin-bottom: 15px; }
  .guide { text-align: left; font-size: 14px; line-height: 1.8; color: #555; background: #fafafa; padding: 15px; border-radius: 8px; }
  .loading { text-align: center; padding: 40px; color: #999; }
  .error { color: #f5222d; padding: 20px; text-align: center; }
</style>
</head>
<body>
<div class="container">
  <h1>📊 股票行情查询工具 (XGBoost 增强版)</h1>
  <div class="search-box">
    <input id="q" placeholder="输入股票代码，如 600519" onkeydown="if(event.key==='Enter')go()">
    <button id="btn" onclick="go()">查 询</button>
  </div>
  <div id="res"></div>
</div>
<script>
const API = "http://127.0.0.1:8000";
async function go(){
  const q = document.getElementById("q").value.trim();
  if(!q) return;
  const btn = document.getElementById("btn");
  btn.disabled = true; btn.textContent = "查询中...";
  document.getElementById("res").innerHTML = '<div class="loading">正在获取并计算数据...</div>';
  try{
    const r = await fetch(`${API}/stock_data?symbol=${q}`);
    const d = await r.json();
    if(d.error){ document.getElementById("res").innerHTML = `<div class="error">${d.error}</div>`; }
    else { render(d); }
  } catch(e){
    document.getElementById("res").innerHTML = '<div class="error">请求失败，请检查后端是否运行。</div>';
  }
  btn.disabled = false; btn.textContent = "查 询";
}
function render(d){
  const rets = d.returns;
  let rh = '';
  for(const [k,v] of Object.entries(rets)){
    let cls = 'neutral'; if(v>0) cls='up'; if(v<0) cls='down';
    let val = v>=0 ? `+${v}%` : `${v}%`;
    rh += `<div class="ret-item"><div class="ret-label">${k}</div><div class="ret-value ${cls}">${val}</div></div>`;
  }
  const pred = d.prediction;
  const dirColor = pred.direction === "看涨" ? "#f5222d" : (pred.direction === "看跌" ? "#52c41a" : "#1890ff");
  
  document.getElementById("res").innerHTML = `
    <div class="card">
      <div class="card-title">📌 ${d.name}（${d.code}）— 最新价 ${d.latest_price} 元</div>
      <p style="font-size:14px;color:#666;line-height:1.8;margin-top:8px;">
        行业：${d.industry} ｜ 上市时间：${d.listing_date}<br>
        板块：${d.industry_performance} <br>
        主营业务：${d.main_business}
      </p>
    </div>
    <div class="card">
      <div class="card-title">📈 各周期涨幅</div>
      <div class="ret-grid">${rh}</div>
    </div>
    <div class="card">
      <div class="card-title">🤖 XGBoost 模型预测</div>
      <div class="pred-box">
        <div class="pred-dir" style="color:${dirColor}">${pred.direction}</div>
        <div class="pred-conf">置信度 ${pred.confidence}% ｜ 历史回测准确率 ${pred.model_accuracy}%</div>
      </div>
      <div class="guide">
        ${pred.guidance.map(g=>'<div>• '+g+'</div>').join('')}
      </div>
    </div>
  `;
}
</script>
</body>
</html>
"""

def get_stock_list():
    global stock_list_cache
    if stock_list_cache is None or stock_list_cache.empty:
        try:
            df = ak.stock_zh_a_spot_em()
            stock_list_cache = df[["代码", "名称"]].drop_duplicates(subset=["代码"])
        except Exception:
            stock_list_cache = pd.DataFrame(columns=["代码", "名称"])
    return stock_list_cache

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content

@app.get("/stock_data")
def stock_data(symbol: str):
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist_em(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
        if df is None or df.empty: raise ValueError
    except:
        return {"error": "获取数据失败，请检查股票代码或网络。"}
    
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
        name = info[info["item"]=="股票简称"]["value"].values[0]
        industry = info[info["item"]=="行业"]["value"].values[0]
        listing = info[info["item"]=="上市时间"]["value"].values[0]
    except:
        name, industry, listing = symbol, "未知", "未知"

    # 获取主营业务
    try:
        # 判断是沪市还是深市，加上对应的前缀
        prefix = "SH" if symbol.startswith("6") else "SZ"
        main_biz_df = ak.stock_zygc_em(symbol=f"{prefix}{symbol}")
        # 获取最新一期“按产品”的主营构成
        main_biz = main_biz_df[main_biz_df["分类类型"]=="按产品分类"]["主营构成"].tolist()
        main_business = "、".join(main_biz[:3]) # 只取前3个核心业务，防止太长
    except:
        main_business = "暂无"

    df = df.rename(columns={"日期": "date", "收盘": "close", "成交量": "volume", "涨跌幅": "pct_change"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    latest = df.iloc[-1]

    # 计算各周期涨幅
    returns = {}
    for name, days in [("近1周", 5), ("近1月", 22), ("近3月", 66), ("近半年", 132), ("近1年", 252)]:
        if len(df) > days:
            base = df.iloc[-(days+1)]["close"]
            returns[name] = round((latest["close"] - base) / base * 100, 2)
        else:
            returns[name] = 0.0
    
    ind_perf = "暂无"
    try:
        bd = ak.stock_board_industry_name_em()
        # 模糊匹配行业名称
        m = bd[bd["板块名称"].str.contains(industry, na=False)]
        if not m.empty:
            board_name = m.iloc[0]["板块名称"]
            bh = ak.stock_board_industry_hist_em(symbol=board_name, period="日k")
            if not bh.empty: 
                latest_pct = bh.iloc[-1]["涨跌幅"]
                ind_perf = f"{board_name} (今日 {latest_pct}%)"
    except: pass

    # XGBoost 预测逻辑
    d = df.copy()
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma60"] = d["close"].rolling(60).mean()
    d["vol_ma5"] = d["volume"].rolling(5).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma5"]
    d["target"] = (d["close"].shift(-5) / d["close"] > 1.03).astype(int)
    d = d.dropna()

    if len(d) < 100:
        return {"error": f"历史数据不足（仅{len(d)}条），无法启动XGBoost模型"}
        
    feats = ["pct_change", "ma5", "ma20", "ma60", "vol_ratio"]
    X, y = d[feats], d["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    mdl = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, eval_metric="logloss", use_label_encoder=False)
    mdl.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, mdl.predict(X_te))
    prob = mdl.predict_proba(X.iloc[-1:].values)[0]
    
    direction = "看涨" if prob[1] > prob[0] else "看跌"
    conf = round(max(prob)*100, 1)
    
    guidance = [
        f"模型预测未来5日趋势：【{direction}】，置信度 {conf}%",
        f"基于 {len(d)} 条历史数据的XGBoost模型，历史回测准确率为 {round(acc*100, 1)}%",
        "⚠️ 模型结果仅供参考，不构成任何投资建议。"
    ]

    result = {
        "code": symbol, "name": name,
        "latest_price": round(float(latest["close"]), 2),
        "listing_date": listing, "industry": industry, "industry_performance": ind_perf,
        "main_business": main_business,
        "returns": returns,
        "prediction": {"direction": direction, "confidence": conf, "model_accuracy": round(acc*100, 1), "guidance": guidance}
    }
    return json.loads(json.dumps(result, default=float))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)