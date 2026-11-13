import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="DEA 專業分析與 AI 智慧顧問系統", layout="wide")

st.title("📊 互動式 DEA 資料包絡分析與 AI 智慧顧問平台")
st.markdown("對標正式 DEA 軟體（MaxDEA / DEAP）標準輸出格式，支援 CCR、BCC、規模報酬（RTS）、鬆弛變數與改善目標、同儕標竿（Peer Group）及 AI 智慧顧問診斷。")

# --- 核心 DEA 線性規劃求解器 (對標正式 DEA 軟體標準格式) ---
def solve_dea_model(df, dmu_col, input_cols, output_cols, orientation="input"):
    """
    同時求解 CCR (CRS) 與 BCC (VRS) 模型（兩階段線性規劃含鬆弛變數），
    計算 SE (規模效率)、RTS (規模報酬狀態)、Targets (目標值)、Slacks (鬆弛變數) 與 Peer Group (同儕群體)。
    """
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
        
        # 1. 求解 CCR (CRS) 效率 (輸入導向第一階段)
        # 變數: [theta, lambda_0, ..., lambda_{n-1}]
        c_ccr = np.zeros(1 + n)
        c_ccr[0] = 1.0  # min theta
        
        # 投入限制: sum(lambda_j * x_ij) - theta * x_ik <= 0
        A_ub_in = np.hstack([-x_k.reshape(-1, 1), X.T])
        b_ub_in = np.zeros(m)
        
        # 產出限制: -sum(lambda_j * y_rj) <= -y_rk
        A_ub_out = np.hstack([np.zeros((s, 1)), -Y.T])
        b_ub_out = -y_k
        
        A_ub = np.vstack([A_ub_in, A_ub_out])
        b_ub = np.concatenate([b_ub_in, b_ub_out])
        bounds_ccr = [(0, None)] * (1 + n)
        
        res_ccr = linprog(c_ccr, A_ub=A_ub, b_ub=b_ub, bounds=bounds_ccr, method='highs')
        ccr_eff = res_ccr.x[0] if res_ccr.success else np.nan
        lambdas_ccr = res_ccr.x[1:] if res_ccr.success else np.zeros(n)
        
        # 2. 求解 BCC (VRS) 效率 (輸入導向第一階段: 加上 sum(lambda) = 1)
        A_eq_bcc = np.zeros((1, 1 + n))
        A_eq_bcc[0, 1:] = 1.0
        b_eq_bcc = np.array([1.0])
        
        res_bcc = linprog(c_ccr, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_bcc, b_eq=b_eq_bcc, bounds=bounds_ccr, method='highs')
        bcc_eff = res_bcc.x[0] if res_bcc.success else np.nan
        
        # 3. 求解 BCC 第二階段 (最大化鬆弛變數 sum(s^-) + sum(s^+)) 以取得精確目標值與 Peer Group
        # 變數: [lambda_0..n-1, s^-_0..m-1, s^+_0..s-1]
        if res_bcc.success:
            c_s2 = np.concatenate([np.zeros(n), -np.ones(m), -np.ones(s)])
            # X^T * lambda + s^- = bcc_eff * x_k
            A_eq_in = np.hstack([X.T, np.eye(m), np.zeros((m, s))])
            b_eq_in = bcc_eff * x_k
            # Y^T * lambda - s^+ = y_k
            A_eq_out = np.hstack([Y.T, np.zeros((s, m)), -np.eye(s)])
            b_eq_out = y_k
            # sum(lambda) = 1
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
            
        # 4. 求解 NIRS (非遞增規模報酬: sum(lambda) <= 1) 以判定規模報酬狀態 (IRS / DRS / CRS)
        A_ub_nirs = np.vstack([A_ub, A_eq_bcc])
        b_ub_nirs = np.concatenate([b_ub, np.array([1.0])])
        res_nirs = linprog(c_ccr, A_ub=A_ub_nirs, b_ub=b_ub_nirs, bounds=bounds_ccr, method='highs')
        nirs_eff = res_nirs.x[0] if res_nirs.success else np.nan
        
        se = ccr_eff / bcc_eff if (bcc_eff and bcc_eff > 1e-6) else np.nan
        
        # 判定規模報酬 (RTS)
        if abs(se - 1.0) < 1e-4:
            rts = "CRS (固定規模報酬)"
        else:
            if abs(bcc_eff - nirs_eff) < 1e-4:
                rts = "DRS (規模報酬遞減)"
            else:
                rts = "IRS (規模報酬遞增)"
                
        # 同儕參考群體 (Peer Group & Lambda Weights)
        peers = []
        for j in range(n):
            if lambdas_bcc[j] > 1e-4:
                peers.append(f"{dmus[j]} ({lambdas_bcc[j]:.3f})")
        peer_str = ", ".join(peers) if peers else "-"
        
        # 計算目標值 (Targets)
        target_in = bcc_eff * x_k - slacks_in
        target_out = y_k + slacks_out
        
        row_dict = {
            "DMU": dmus[k],
            "CCR_TE (整體技術效率)": round(ccr_eff, 4),
            "BCC_PTE (純技術效率)": round(bcc_eff, 4),
            "SE (規模效率)": round(se, 4),
            "RTS (規模報酬狀態)": rts,
            "Peer_Group (同儕標竿與權重)": peer_str
        }
        
        for i, col in enumerate(input_cols):
            row_dict[f"{col}_原始值"] = x_k[i]
            row_dict[f"{col}_目標值"] = round(target_in[i], 4)
            row_dict[f"{col}_鬆弛變數(Slack)"] = round(slacks_in[i], 4)
            
        for r, col in enumerate(output_cols):
            row_dict[f"{col}_原始值"] = y_k[r]
            row_dict[f"{col}_目標值"] = round(target_out[r], 4)
            row_dict[f"{col}_鬆弛變數(Slack)"] = round(slacks_out[r], 4)
            
        results.append(row_dict)
        
    return pd.DataFrame(results)

# --- 側邊欄：動態參數設定 ---
st.sidebar.header("⚙️ 參數設定與模型選擇")
uploaded_file = st.sidebar.file_uploader("上傳 Excel 資料檔 (.xlsx)", type=["xlsx"])

# 提供範例資料下載，方便使用者直接測試
sample_data = pd.DataFrame({
    "Country": ["Austria", "Belgium", "Denmark", "Finland", "France", "Germany", "Italy", "Netherlands", "Spain", "Sweden"],
    "Capital": [120, 150, 95, 85, 680, 920, 510, 240, 380, 160],
    "Labor": [4.3, 4.8, 2.9, 2.5, 29.5, 43.2, 23.1, 8.9, 19.8, 5.1],
    "Energy": [34, 52, 18, 32, 240, 310, 155, 78, 125, 48],
    "GDP": [450, 530, 360, 275, 2780, 3850, 2010, 910, 1420, 540]
})
sample_buffer = io.BytesIO()
with pd.ExcelWriter(sample_buffer, engine='openpyxl') as writer:
    sample_data.to_excel(writer, index=False)

st.sidebar.download_button(
    label="📄 下載測試用範例 Excel",
    data=sample_buffer.getvalue(),
    file_name="sample_dea_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ 檔案讀取成功！")
    
    columns = list(df.columns)
    dmu_col = st.sidebar.selectbox("1. 選擇 DMU（決策單元名稱）欄位", columns, index=0)
    numeric_cols = [c for c in columns if c != dmu_col and pd.api.types.is_numeric_dtype(df[c])]
    
    input_cols = st.sidebar.multiselect("2. 選擇 投入變數 (Inputs)", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
    remaining_cols = [c for c in numeric_cols if c not in input_cols]
    output_cols = st.sidebar.multiselect("3. 選擇 產出變數 (Outputs)", remaining_cols, default=remaining_cols[:min(1, len(remaining_cols))])
    
    model_type = st.sidebar.selectbox("4. 選擇分析模式", ["CCR + BCC 綜合效率與規模報酬分析"])
    
    if st.sidebar.button("🚀 執行 DEA 專業分析", type="primary"):
        if not input_cols or not output_cols:
            st.error("⚠️ 請至少選擇一個投入變數與一個產出變數！")
        elif df[input_cols + output_cols].isnull().any().any():
            st.error("⚠️ 偵測到所選變數中包含遺漏值 (NaN)，請先清理資料或刪除缺漏列後再試！")
        else:
            with st.spinner("正在進行線性規劃求解與報表生成..."):
                res_df = solve_dea_model(df, dmu_col, input_cols, output_cols)
            
            # 拆分為正式 DEA 軟體標準四大報表
            eff_cols = ["DMU", "CCR_TE (整體技術效率)", "BCC_PTE (純技術效率)", "SE (規模效率)", "RTS (規模報酬狀態)"]
            df_eff = res_df[eff_cols]
            
            target_cols = ["DMU"] + [c for c in res_df.columns if "_原始值" in c or "_目標值" in c or "_鬆弛變數" in c]
            df_targets = res_df[target_cols]
            
            peer_cols = ["DMU", "BCC_PTE (純技術效率)", "Peer_Group (同儕標竿與權重)"]
            df_peers = res_df[peer_cols]
            
            # --- 網頁分頁呈現 ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 1. 效率與規模報酬總表", 
                "🎯 2. 目標值與鬆弛變數", 
                "🤝 3. 同儕標竿群體 (Peer Group)", 
                "📊 4. Plotly 互動視覺化", 
                "🤖 5. AI 智慧顧問診斷與策略"
            ])
            
            with tab1:
                st.subheader("技術效率分解與規模報酬狀態表 (CCR / BCC / SE / RTS)")
                st.dataframe(df_eff, use_container_width=True)
                
            with tab2:
                st.subheader("潛在改善目標與鬆弛變數分析 (Targets & Slacks)")
                st.dataframe(df_targets, use_container_width=True)
                
            with tab3:
                st.subheader("同儕參考群體與對標權重 (Peer Reference Set)")
                st.dataframe(df_peers, use_container_width=True)
                
            with tab4:
                st.subheader("各決策單元效率比較與規模報酬分布")
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=df_eff["DMU"], y=df_eff["CCR_TE (整體技術效率)"], name="CCR 整體技術效率"))
                fig_bar.add_trace(go.Bar(x=df_eff["DMU"], y=df_eff["BCC_PTE (純技術效率)"], name="BCC 純技術效率"))
                fig_bar.add_trace(go.Scatter(x=df_eff["DMU"], y=df_eff["SE (規模效率)"], name="規模效率 (SE)", mode="lines+markers"))
                fig_bar.update_layout(barmode="group", title="CCR、BCC 與規模效率 (SE) 綜合對照圖", yaxis_range=[0, 1.1])
                st.plotly_chart(fig_bar, use_container_width=True)
                
                rts_counts = df_eff["RTS (規模報酬狀態)"].value_counts().reset_index()
                rts_counts.columns = ["規模報酬狀態", "單位數量"]
                fig_pie = px.pie(rts_counts, names="規模報酬狀態", values="單位數量", title="受評單位規模報酬 (RTS) 結構分布")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with tab5:
                st.subheader("🤖 AI 智慧顧問：模型診斷、結果解讀與策略改善建議")
                avg_ccr = df_eff["CCR_TE (整體技術效率)"].mean()
                avg_bcc = df_eff["BCC_PTE (純技術效率)"].mean()
                avg_se = df_eff["SE (規模效率)"].mean()
                eff_dmus = df_eff[df_eff["CCR_TE (整體技術效率)"] >= 0.9999]["DMU"].tolist()
                ineff_df = df_eff[df_eff["BCC_PTE (純技術效率)"] < 0.9999]
                
                # 檢查是否有潛在不良產出關鍵字
                bad_keywords = ["co2", "ghg", "emission", "waste", "pollution", "溫室氣體", "碳", "汙染", "廢棄"]
                has_bad_output = any(any(k in col.lower() for k in bad_keywords) for col in output_cols)
                
                st.markdown("#### 1️⃣ 模型適配性診斷 (Model Diagnosis)")
                if has_bad_output:
                    st.warning("⚠️ **偵測到潛在「不良產出（Undesirable Output）」變數**：您選擇的產出欄位中包含碳排放或汙染物相關字眼。傳統 CCR/BCC 假設產出愈多愈好，建議在正式論文中進一步採用 **SBM 不良產出模型** 或 **方向性距離函數 (DDF)** 以避免環境效率高估。")
                else:
                    st.info(f"✅ **資料與模型檢核通過**：本次共評估 **{len(df)}** 個決策單元（DMU），包含 **{len(input_cols)}** 項投入與 **{len(output_cols)}** 項期望產出。樣本數符合經驗法則（DMU 數量 ≥ 投入與產出總和之 2~3 倍），線性規劃求解收斂穩定。")
                
                st.markdown("#### 2️⃣ 整體效率與規模解讀 (Automated Interpretation)")
                st.write(f"""
                * **平均表現**：全體單位之平均 CCR 整體技術效率為 **{avg_ccr:.4f}**，平均 BCC 純技術效率為 **{avg_bcc:.4f}**，平均規模效率（SE）為 **{avg_se:.4f}**。
                * **效率前緣標竿**：達到 CCR 整體效率前緣（分數 = 1.0000）的標竿單位共有 **{len(eff_dmus)}** 個：{', '.join(eff_dmus) if eff_dmus else '無'}。
                * **效率落差主因**：{'純技術效率（管理與資源配置能力）偏低是造成整體無效率的主要原因' if avg_bcc < avg_se else '規模效率（營運規模未達最適水準）偏低是造成整體無效率的主要原因'}。
                """)
                
                st.markdown("#### 3️⃣ 策略改善與對標建議 (Actionable Insights)")
                if len(ineff_df) > 0:
                    worst_row = ineff_df.sort_values("BCC_PTE (純技術效率)").iloc[0]
                    w_dmu = worst_row["DMU"]
                    w_peer = res_df[res_df["DMU"] == w_dmu]["Peer_Group (同儕標竿與權重)"].values[0]
                    st.success(f"""
                    * **優先改善對象**：建議優先關注純技術效率最低的 **{w_dmu}**（BCC 效率值 = {worst_row['BCC_PTE (純技術效率)']:.4f}）。
                    * **標竿學習方向**：該單位應參考其同儕參考群體 **{w_peer}** 之投入產出配置比例，並依據「分頁 2：目標值與鬆弛變數」進行投入要素削減。
                    * **規模調整策略**：針對處於 **IRS（規模報酬遞增）** 的單位，建議適度擴大投入規模以獲取規模經濟效益；處於 **DRS（規模報酬遞減）** 的單位則應避免盲目擴張。
                    """)
                else:
                    st.success("🎉 所有受評單位在 BCC 模型下皆已達到純技術效率前緣！")

            # --- 一鍵匯出多工作表 Excel 專業報表 ---
            st.divider()
            excel_out = io.BytesIO()
            with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
                df_eff.to_excel(writer, index=False, sheet_name='1_效率與規模報酬總表')
                df_targets.to_excel(writer, index=False, sheet_name='2_目標值與鬆弛變數')
                df_peers.to_excel(writer, index=False, sheet_name='3_同儕標竿群體')
                df.to_excel(writer, index=False, sheet_name='4_原始輸入資料')
                
            st.download_button(
                label="📥 一鍵下載完整 DEA 專業分析報表 (.xlsx)",
                data=excel_out.getvalue(),
                file_name="DEA_Professional_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
else:
    st.info("👈 請先在左側側邊欄上傳您的 Excel 檔案（或點擊「下載測試用範例 Excel」體驗系統功能）。")
