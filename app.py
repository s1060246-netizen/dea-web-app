import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px
import plotly.graph_objects as go
import io
import re

st.set_page_config(page_title="DEA 跨年度與 AI 智慧顧問系統", layout="wide")

st.title("📊 互動式 DEA 跨年度分析與 AI 智慧顧問平台")
st.markdown("支援**多年度獨立 Excel 檔案批次上傳**（如 2019–2023 年五個檔案），自動對齊決策單元（DMU），完整產出 **單年度 CCR / BCC / SE / RTS / Targets / Peer Group** 與 **跨期 Malmquist 生產力指數（TFP、EC、TC）**！")

# --- 1. 單年度 DEA 核心求解器 (CCR, BCC, SE, RTS, Targets, Slacks, Peer Group) ---
def solve_single_year_dea(df, dmu_col, input_cols, output_cols):
    n = len(df)
    m = len(input_cols)
    s = len(output_cols)
    
    X = df[input_cols].values.astype(float)
    Y = df[output_cols].values.astype(float)
    dmus = df[dmu_col].astype(str).tolist()
    
    results = []
    
    for k in range(n):
        x_k = X[k]
        y_k = Y[k]
        
        # CCR (CRS)
        c_ccr = np.zeros(1 + n)
        c_ccr[0] = 1.0
        
        A_ub_in = np.hstack([-x_k.reshape(-1, 1), X.T])
        b_ub_in = np.zeros(m)
        A_ub_out = np.hstack([np.zeros((s, 1)), -Y.T])
        b_ub_out = -y_k
        
        A_ub = np.vstack([A_ub_in, A_ub_out])
        b_ub = np.concatenate([b_ub_in, b_ub_out])
        bounds_ccr = [(0, None)] * (1 + n)
        
        res_ccr = linprog(c_ccr, A_ub=A_ub, b_ub=b_ub, bounds=bounds_ccr, method='highs')
        ccr_eff = res_ccr.x[0] if res_ccr.success else np.nan
        
        # BCC (VRS) 第一階段
        A_eq_bcc = np.zeros((1, 1 + n))
        A_eq_bcc[0, 1:] = 1.0
        b_eq_bcc = np.array([1.0])
        
        res_bcc = linprog(c_ccr, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_bcc, b_eq=b_eq_bcc, bounds=bounds_ccr, method='highs')
        bcc_eff = res_bcc.x[0] if res_bcc.success else np.nan
        
        # BCC 第二階段（鬆弛變數與同儕群體）
        if res_bcc.success:
            c_s2 = np.concatenate([np.zeros(n), -np.ones(m), -np.ones(s)])
            A_eq_in = np.hstack([X.T, np.eye(m), np.zeros((m, s))])
            b_eq_in = bcc_eff * x_k
            A_eq_out = np.hstack([Y.T, np.zeros((s, m)), -np.eye(s)])
            b_eq_out = y_k
            A_eq_vrs = np.hstack([np.ones((1, n)), np.zeros((1, m + s))])
            b_eq_vrs = np.array([1.0])
            
            A_eq_s2 = np.vstack([A_eq_in, A_eq_out, A_eq_vrs])
            b_eq_s2 = np.concatenate([b_eq_in, b_eq_out, b_eq_vrs])
            bounds_s2 = [(0, None)] * (n + m + s)
            
            res_s2 = linprog(c_s2, A_eq=A_eq_s2, b_eq=b_eq_s2, bounds=bounds_s2, method='highs')
            if res_s2.success:
                lambdas_bcc = res_s2.x[:n]
                slacks_in = res_s2.x[n:n+m]
                slacks_out = res_s2.x[n+m:]
            else:
                lambdas_bcc = res_bcc.x[1:]
                slacks_in = np.zeros(m)
                slacks_out = np.zeros(s)
        else:
            lambdas_bcc = np.zeros(n)
            slacks_in = np.zeros(m)
            slacks_out = np.zeros(s)
            
        # NIRS 判定規模報酬狀態
        A_ub_nirs = np.vstack([A_ub, A_eq_bcc])
        b_ub_nirs = np.concatenate([b_ub, np.array([1.0])])
        res_nirs = linprog(c_ccr, A_ub=A_ub_nirs, b_ub=b_ub_nirs, bounds=bounds_ccr, method='highs')
        nirs_eff = res_nirs.x[0] if res_nirs.success else np.nan
        
        se = ccr_eff / bcc_eff if (bcc_eff and bcc_eff > 1e-6) else np.nan
        
        if abs(se - 1.0) < 1e-4:
            rts = "CRS (固定規模報酬)"
        else:
            if abs(bcc_eff - nirs_eff) < 1e-4:
                rts = "DRS (規模報酬遞減)"
            else:
                rts = "IRS (規模報酬遞增)"
                
        peers = [f"{dmus[j]} ({lambdas_bcc[j]:.3f})" for j in range(n) if lambdas_bcc[j] > 1e-4]
        peer_str = ", ".join(peers) if peers else "-"
        
        target_in = bcc_eff * x_k - slacks_in
        target_out = y_k + slacks_out
        
        row_dict = {
            "DMU": dmus[k],
            "CCR_TE": round(ccr_eff, 4),
            "BCC_PTE": round(bcc_eff, 4),
            "SE": round(se, 4),
            "RTS": rts,
            "Peer_Group": peer_str
        }
        
        for i, col in enumerate(input_cols):
            row_dict[f"{col}_原始值"] = x_k[i]
            row_dict[f"{col}_目標值"] = round(target_in[i], 4)
            row_dict[f"{col}_Slack"] = round(slacks_in[i], 4)
            
        for r, col in enumerate(output_cols):
            row_dict[f"{col}_原始值"] = y_k[r]
            row_dict[f"{col}_目標值"] = round(target_out[r], 4)
            row_dict[f"{col}_Slack"] = round(slacks_out[r], 4)
            
        results.append(row_dict)
        
    return pd.DataFrame(results)

# --- 2. 跨期距離函數與 Malmquist 生產力指數計算 (CRS 輸入導向) ---
def compute_distance_function(X_eval, Y_eval, X_ref, Y_ref):
    """計算 D_ref(x_eval, y_eval)：以 ref 時期為技術前緣評估 eval 時期單位"""
    n_eval = X_eval.shape[0]
    n_ref = X_ref.shape[0]
    m = X_eval.shape[1]
    s = Y_eval.shape[1]
    scores = []
    
    for k in range(n_eval):
        x_k = X_eval[k]
        y_k = Y_eval[k]
        
        c = np.zeros(1 + n_ref)
        c[0] = 1.0
        
        A_ub_in = np.hstack([-x_k.reshape(-1, 1), X_ref.T])
        b_ub_in = np.zeros(m)
        A_ub_out = np.hstack([np.zeros((s, 1)), -Y_ref.T])
        b_ub_out = -y_k
        
        A_ub = np.vstack([A_ub_in, A_ub_out])
        b_ub = np.concatenate([b_ub_in, b_ub_out])
        bounds = [(0, None)] * (1 + n_ref)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        scores.append(res.x[0] if res.success else np.nan)
        
    return np.array(scores)

def solve_malmquist(aligned_dfs, years, dmu_col, input_cols, output_cols):
    """計算相鄰年度之間的 Malmquist 指數 (EC, TC, TFPCH)"""
    malmquist_records = []
    
    for idx in range(len(years) - 1):
        t1 = years[idx]
        t2 = years[idx + 1]
        period_label = f"{t1} → {t2}"
        
        df1 = aligned_dfs[t1]
        df2 = aligned_dfs[t2]
        
        X1 = df1[input_cols].values.astype(float)
        Y1 = df1[output_cols].values.astype(float)
        X2 = df2[input_cols].values.astype(float)
        Y2 = df2[output_cols].values.astype(float)
        dmus = df1[dmu_col].astype(str).tolist()
        
        # D^t(x^t, y^t)
        d_t_t = compute_distance_function(X1, Y1, X1, Y1)
        # D^{t+1}(x^{t+1}, y^{t+1})
        d_t1_t1 = compute_distance_function(X2, Y2, X2, Y2)
        # D^t(x^{t+1}, y^{t+1})
        d_t_t1 = compute_distance_function(X2, Y2, X1, Y1)
        # D^{t+1}(x^t, y^t)
        d_t1_t = compute_distance_function(X1, Y1, X2, Y2)
        
        # 效率變動 (EC = Catch-up)
        ec = d_t1_t1 / d_t_t
        # 技術變動 (TC = Frontier-shift)
        tc = np.sqrt((d_t_t1 / d_t1_t1) * (d_t_t / d_t1_t))
        # 總要素生產力變動 (TFPCH = EC * TC)
        tfpch = ec * tc
        
        for k in range(len(dmus)):
            malmquist_records.append({
                "期間 (Period)": period_label,
                "DMU": dmus[k],
                "效率變動 (EC)": round(ec[k], 4),
                "技術變動 (TC)": round(tc[k], 4),
                "總要素生產力變動 (TFPCH)": round(tfpch[k], 4),
                "生產力狀態": "成長 (>1)" if tfpch[k] > 1.0001 else ("衰退 (<1)" if tfpch[k] < 0.9999 else "持平 (=1)")
            })
            
    return pd.DataFrame(malmquist_records)

# --- 側邊欄：多檔案批次上傳與動態參數設定 ---
st.sidebar.header("📂 1. 多年度檔案上傳 (方案 B)")
uploaded_files = st.sidebar.file_uploader(
    "請一次選取多個年度的 Excel 檔案（例如 2019~2023 共 5 個檔案）", 
    type=["xlsx"], 
    accept_multiple_files=True
)

# 提供 2019~2023 測試用 ZIP 或單檔提示
st.sidebar.caption("💡 提示：檔名建議包含年份（如 `data2019.xlsx`、`data2020.xlsx`），系統會自動辨識年份並排序。")

if uploaded_files:
    # 依檔名中的年份自動排序檔案
    file_year_map = {}
    for idx, f in enumerate(uploaded_files):
        match = re.search(r'(20\d{2}|19\d{2})', f.name)
        year_label = match.group(1) if match else f"檔案_{idx+1}_{f.name}"
        file_year_map[year_label] = f
        
    sorted_years = sorted(file_year_map.keys())
    st.sidebar.success(f"✅ 已載入 **{len(sorted_years)}** 個年度檔案：{', '.join(sorted_years)}")
    
    # 讀取第一個檔案作為欄位設定基準
    sample_df = pd.read_excel(file_year_map[sorted_years[0]])
    columns = list(sample_df.columns)
    
    st.sidebar.header("⚙️ 2. 變數與模型設定")
    dmu_col = st.sidebar.selectbox("選擇 DMU（單位名稱/國家）欄位", columns, index=0)
    numeric_cols = [c for c in columns if c != dmu_col and pd.api.types.is_numeric_dtype(sample_df[c])]
    
    # 預設自動抓出常見投入產出名稱（若無則取前幾欄）
    default_inputs = [c for c in numeric_cols if "(i)" in c.lower() or "input" in c.lower() or "資本" in c or "勞動" in c or "能源" in c]
    if not default_inputs:
        default_inputs = numeric_cols[:min(3, len(numeric_cols))]
        
    remaining_cols = [c for c in numeric_cols if c not in default_inputs]
    default_outputs = [c for c in remaining_cols if "(o)" in c.lower() or "output" in c.lower() or "gdp" in c.lower() or "溫室" in c]
    if not default_outputs:
        default_outputs = remaining_cols[:min(1, len(remaining_cols))]
        
    input_cols = st.sidebar.multiselect("選擇 投入變數 (Inputs)", numeric_cols, default=default_inputs)
    output_cols = st.sidebar.multiselect("選擇 產出變數 (Outputs)", [c for c in numeric_cols if c not in input_cols], default=default_outputs)
    
    if st.sidebar.button("🚀 執行跨年度 DEA 與 Malmquist 完整分析", type="primary"):
        if not input_cols or not output_cols:
            st.error("⚠️ 請至少選擇一個投入變數與一個產出變數！")
        else:
            with st.spinner("正在對齊多年度決策單元並進行線性規劃求解..."):
                raw_dfs = {}
                valid_dmu_sets = []
                
                for yr in sorted_years:
                    df_yr = pd.read_excel(file_year_map[yr])
                    # 確保必要欄位存在
                    missing_cols = [c for c in [dmu_col] + input_cols + output_cols if c not in df_yr.columns]
                    if missing_cols:
                        st.error(f"❌ 檔案 `{file_year_map[yr].name}` 缺少欄位：{missing_cols}，請確認各年度檔案欄位名稱一致！")
                        st.stop()
                        
                    df_clean = df_yr[[dmu_col] + input_cols + output_cols].dropna().copy()
                    df_clean[dmu_col] = df_clean[dmu_col].astype(str).str.strip()
                    raw_dfs[yr] = df_clean
                    valid_dmu_sets.append(set(df_clean[dmu_col]))
                
                # 取所有年度共同擁有的平衡面板 (Balanced Panel) DMU 並依名稱排序
                common_dmus = sorted(list(set.intersection(*valid_dmu_sets)))
                
                if len(common_dmus) < 3:
                    st.error("❌ 各年度共同擁有的有效單位（DMU）不足 3 個，無法進行跨年度對齊分析，請檢查各檔案中的單位名稱是否一致。")
                    st.stop()
                    
                aligned_dfs = {}
                for yr in sorted_years:
                    df_sub = raw_dfs[yr][raw_dfs[yr][dmu_col].isin(common_dmus)].sort_values(by=dmu_col).reset_index(drop=True)
                    aligned_dfs[yr] = df_sub
                    
                # 計算各年度單期完整 DEA 報表
                yearly_dea_results = {}
                summary_ccr = pd.DataFrame({"DMU": common_dmus})
                summary_bcc = pd.DataFrame({"DMU": common_dmus})
                summary_se = pd.DataFrame({"DMU": common_dmus})
                summary_rts = pd.DataFrame({"DMU": common_dmus})
                
                for yr in sorted_years:
                    res_yr = solve_single_year_dea(aligned_dfs[yr], dmu_col, input_cols, output_cols)
                    yearly_dea_results[yr] = res_yr
                    summary_ccr[f"CCR_{yr}"] = res_yr["CCR_TE"]
                    summary_bcc[f"BCC_{yr}"] = res_yr["BCC_PTE"]
                    summary_se[f"SE_{yr}"] = res_yr["SE"]
                    summary_rts[f"RTS_{yr}"] = res_yr["RTS"]
                    
                # 計算跨期 Malmquist 生產力指數（若上傳 >= 2 個年度）
                has_malmquist = len(sorted_years) >= 2
                if has_malmquist:
                    df_malmquist = solve_malmquist(aligned_dfs, sorted_years, dmu_col, input_cols, output_cols)
                    # 計算各單位跨期幾何平均 (Geometric Mean)
                    def geom_mean(x):
                        return np.exp(np.mean(np.log(x[x > 0])))
                    df_malm_mean = df_malmquist.groupby("DMU", as_index=False).agg({
                        "效率變動 (EC)": geom_mean,
                        "技術變動 (TC)": geom_mean,
                        "總要素生產力變動 (TFPCH)": geom_mean
                    }).round(4)
                    df_malm_mean["整體生產力趨勢"] = df_malm_mean["總要素生產力變動 (TFPCH)"].apply(
                        lambda v: "整體進步 (>1)" if v > 1.0001 else ("整體退步 (<1)" if v < 0.9999 else "持平 (=1)")
                    )

            st.success(f"🎉 分析完成！已自動對齊 **{len(common_dmus)}** 個跨年度共同決策單元（{sorted_years[0]} ~ {sorted_years[-1]}）。")
            
            # --- 網頁五大專業分頁呈現 ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 1. 跨年度效率與規模報酬總表",
                "🚀 2. Malmquist 跨期生產力指數",
                "🎯 3. 單年度明細（Targets & Peers）",
                "📊 4. Plotly 動態趨勢與比較圖表",
                "🤖 5. AI 智慧顧問跨期診斷與策略"
            ])
            
            with tab1:
                st.subheader("多年度 CCR 整體技術效率總表")
                st.dataframe(summary_ccr, use_container_width=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("多年度 BCC 純技術效率總表")
                    st.dataframe(summary_bcc, use_container_width=True)
                with col_b:
                    st.subheader("多年度規模效率 (SE) 總表")
                    st.dataframe(summary_se, use_container_width=True)
                    
                st.subheader("多年度規模報酬狀態 (RTS) 演變表")
                st.dataframe(summary_rts, use_container_width=True)
                
            with tab2:
                if has_malmquist:
                    st.subheader(f"各決策單元跨期幾何平均生產力指數 ({sorted_years[0]}–{sorted_years[-1]})")
                    st.dataframe(df_malm_mean, use_container_width=True)
                    
                    st.subheader("相鄰年度 Malmquist 生產力變動明細 (EC / TC / TFPCH)")
                    st.dataframe(df_malmquist, use_container_width=True)
                else:
                    st.info("請至少上傳 2 個年度的檔案以計算跨期 Malmquist 生產力指數。")
                    
            with tab3:
                selected_yr = st.selectbox("選擇欲檢視明細之年度：", sorted_years, index=len(sorted_years)-1)
                st.subheader(f"{selected_yr} 年度：效率分數、鬆弛變數、目標值與同儕標竿 (Peer Group)")
                st.dataframe(yearly_dea_results[selected_yr], use_container_width=True)
                
            with tab4:
                st.subheader("各單位跨年度 CCR 效率軌跡折線圖")
                ccr_long = summary_ccr.melt(id_vars="DMU", var_name="年度", value_name="CCR_效率值")
                ccr_long["年度"] = ccr_long["年度"].str.replace("CCR_", "")
                fig_trend = px.line(ccr_long, x="年度", y="CCR_效率值", color="DMU", markers=True, title="2019–2023 各決策單元 CCR 技術效率變化軌跡")
                st.plotly_chart(fig_trend, use_container_width=True)
                
                if has_malmquist:
                    st.subheader("跨期平均總要素生產力分解圖 (EC 追趕效應 vs. TC 技術進步)")
                    fig_malm = go.Figure()
                    fig_malm.add_trace(go.Bar(x=df_malm_mean["DMU"], y=df_malm_mean["效率變動 (EC)"], name="效率變動 (EC)"))
                    fig_malm.add_trace(go.Bar(x=df_malm_mean["DMU"], y=df_malm_mean["技術變動 (TC)"], name="技術變動 (TC)"))
                    fig_malm.add_trace(go.Scatter(x=df_malm_mean["DMU"], y=df_malm_mean["總要素生產力變動 (TFPCH)"], name="總要素生產力 (TFPCH)", mode="lines+markers"))
                    fig_malm.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="基準線 (1.0)")
                    fig_malm.update_layout(barmode="group", title="各單位跨期平均 Malmquist 指數分解對照圖")
                    st.plotly_chart(fig_malm, use_container_width=True)
                    
            with tab5:
                st.subheader("🤖 AI 智慧顧問：跨年度面板診斷、解讀與策略改善")
                bad_keywords = ["co2", "ghg", "emission", "waste", "pollution", "溫室氣體", "碳", "汙染", "廢棄"]
                has_bad_output = any(any(k in col.lower() for k in bad_keywords) for col in output_cols)
                
                st.markdown("#### 1️⃣ 跨年度面板與模型適配診斷")
                if has_bad_output:
                    st.warning("⚠️ **偵測到潛在「不良產出（Undesirable Output）」變數**：您的產出欄位中包含溫室氣體或碳排放字眼。在跨年度環境生產力分析中，若已進行資料平移（如 `+ 平移`），請注意傳統徑向模型在平移後可能具有單位不變性限制；建議在學術論文討論中補充說明，或進一步對照 **Malmquist-Luenberger (ML) 指數**。")
                else:
                    st.info(f"✅ **平衡面板對齊完成**：系統已自動從 {len(sorted_years)} 個檔案中篩選出 **{len(common_dmus)}** 個跨年度共同樣本，投入變數 {len(input_cols)} 項、產出變數 {len(output_cols)} 項，樣本結構完整且求解穩定。")
                    
                if has_malmquist:
                    avg_tfp = df_malm_mean["總要素生產力變動 (TFPCH)"].mean()
                    avg_ec = df_malm_mean["效率變動 (EC)"].mean()
                    avg_tc = df_malm_mean["技術變動 (TC)"].mean()
                    top_dmu = df_malm_mean.sort_values("總要素生產力變動 (TFPCH)", ascending=False).iloc[0]
                    bottom_dmu = df_malm_mean.sort_values("總要素生產力變動 (TFPCH)", ascending=True).iloc[0]
                    
                    st.markdown("#### 2️⃣ 跨期生產力演變核心解讀")
                    st.write(f"""
                    * **整體生產力趨勢**：在 **{sorted_years[0]} 至 {sorted_years[-1]}** 年間，全體受評單位的平均總要素生產力（TFPCH）為 **{avg_tfp:.4f}**（{'整體呈現正向成長' if avg_tfp > 1 else '整體呈現微幅衰退'}）。
                    * **驅動引擎拆解**：平均效率變動（EC，追趕效應）為 **{avg_ec:.4f}**，平均技術變動（TC，前緣移動）為 **{avg_tc:.4f}**。顯示這段期間生產力變化的主要推動力來自於 **{'技術前緣向外擴張（技術創新）' if avg_tc > avg_ec else '無效率單位向標竿追趕（管理與配置改善）'}**。
                    * **表現最佳標竿**：跨期生產力成長最強勁的單位為 **{top_dmu['DMU']}**（平均 TFPCH = **{top_dmu['總要素生產力變動 (TFPCH)']:.4f}**）。
                    """)
                    
                    st.markdown("#### 3️⃣ 針對弱勢單位之策略改善方針")
                    latest_yr = sorted_years[-1]
                    latest_peer = yearly_dea_results[latest_yr][yearly_dea_results[latest_yr]["DMU"] == bottom_dmu["DMU"]]["Peer_Group"].values[0]
                    st.success(f"""
                    * **重點關注對象**：**{bottom_dmu['DMU']}** 在此期間的總要素生產力表現相對落後（平均 TFPCH = **{bottom_dmu['總要素生產力變動 (TFPCH)']:.4f}**）。
                    * **診斷與行動建議**：該單位在 {latest_yr} 年度的同儕標竿群體為 **{latest_peer}**。建議檢視「分頁 3」中該單位在能源消耗量、資本與勞動力上的鬆弛變數（Slack）冗餘幅度，並依據「分頁 1」之規模報酬狀態（IRS/DRS）調整資源投入規模。
                    """)

            # --- 一鍵匯出完整跨年度多工作表 Excel 報表 ---
            st.divider()
            excel_out = io.BytesIO()
            with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
                summary_ccr.to_excel(writer, index=False, sheet_name='1_跨年度CCR效率總表')
                summary_bcc.to_excel(writer, index=False, sheet_name='2_跨年度BCC效率總表')
                summary_se.to_excel(writer, index=False, sheet_name='3_跨年度規模效率SE')
                summary_rts.to_excel(writer, index=False, sheet_name='4_跨年度規模報酬RTS')
                if has_malmquist:
                    df_malm_mean.to_excel(writer, index=False, sheet_name='5_Malmquist跨期平均')
                    df_malmquist.to_excel(writer, index=False, sheet_name='6_Malmquist逐期明細')
                for yr in sorted_years:
                    yearly_dea_results[yr].to_excel(writer, index=False, sheet_name=f'{yr}_單年度完整明細')
                    
            st.download_button(
                label=f"📥 一鍵下載 {sorted_years[0]}–{sorted_years[-1]} 完整跨年度 DEA 專業報表 (.xlsx)",
                data=excel_out.getvalue(),
                file_name=f"DEA_MultiYear_Report_{sorted_years[0]}_{sorted_years[-1]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
else:
    st.info("👈 請在左側側邊欄一次選取並上傳您的多個年度 Excel 檔案（例如 `data2019.xlsx` 至 `data2023.xlsx` 共 5 個檔案）。")
