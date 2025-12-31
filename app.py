import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Surgery Center Analytics: Multi-Site", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 10px;
        border-radius: 5px;
        color: #31333F;
    }
    .center-header {
        text-align: center;
        font-weight: bold;
        padding: 5px;
        margin-bottom: 10px;
        border-bottom: 2px solid #ccc;
    }
    .parameter-box {
        background-color: #0E1117; 
        border: 1px solid #262730;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & Processing ---
@st.cache_data
def load_and_process_data():
    # Define file mappings
    files = {
        'ELW': {'ship': 'raw_shipments_ELW.csv', 'bill': 'raw_billing_ELW.csv'},
        'OCNH': {'ship': 'raw_shipments_OCNH.csv', 'bill': 'raw_billing_OCNH.csv'}
    }
    
    shipments_list = []
    billing_list = []
    
    # Load and tag data
    for center, paths in files.items():
        try:
            # Load Shipments
            s_df = pd.read_csv(paths['ship'])
            s_df['center_id'] = center 
            shipments_list.append(s_df)
            
            # Load Billing
            b_df = pd.read_csv(paths['bill'])
            b_df['center_id'] = center
            billing_list.append(b_df)
            
        except FileNotFoundError:
            continue # Skip if file not found, allowing partial loading

    if not shipments_list or not billing_list:
        st.error("Data files missing. Please ensure 'raw_shipments_[CENTER].csv' and 'raw_billing_[CENTER].csv' exist for ELW and OCNH.")
        return None, None, None, None

    # Combine Data
    shipments_df = pd.concat(shipments_list, ignore_index=True)
    billing_df = pd.concat(billing_list, ignore_index=True)

    # Date Conversion
    date_cols_ship = ['order_date', 'ship_date', 'delivery_date']
    for col in date_cols_ship:
        shipments_df[col] = pd.to_datetime(shipments_df[col], dayfirst=True)

    date_cols_bill = ['use_date', 'order_date', 'delivery_date']
    for col in date_cols_bill:
        billing_df[col] = pd.to_datetime(billing_df[col], dayfirst=True)

    # --- Financial Logic ---
    pricing = {
        'PC580': {'ASP': 140, 'COGS': 50},
        'PC545': {'ASP': 90, 'COGS': 30}
    }

    # Revenue & COGS
    def get_financials(row):
        model = row['model']
        asp = pricing.get(model, {}).get('ASP', 0)
        cogs = pricing.get(model, {}).get('COGS', 0)
        return pd.Series([asp, cogs])

    billing_df[['Revenue', 'COGS']] = billing_df.apply(get_financials, axis=1)
    billing_df['Gross_Margin'] = billing_df['Revenue'] - billing_df['COGS']

    # Freight Logic
    # Group by center_id AND order_number to avoid mixing orders
    shipment_events = shipments_df[['center_id', 'order_number', 'order_date', 'delivery_date', 'freight_mode']].drop_duplicates()
    
    def calculate_freight(row):
        days_diff = (row['delivery_date'] - row['order_date']).days
        if days_diff == 0:
            return 120 # Same Day
        elif days_diff == 1:
            return 50  # Next Day
        else:
            return 25  # Standard

    shipment_events['Freight_Cost'] = shipment_events.apply(calculate_freight, axis=1)

    # --- Inventory Logic ---
    # We now handle ELW and OCNH differently due to different stock data sources
    
    # 1. ELW Inventory (Calculated from Shipments - Billing)
    ship_elw = shipments_df[shipments_df['center_id'] == 'ELW']
    bill_elw = billing_df[billing_df['center_id'] == 'ELW']
    
    s_qty_elw = ship_elw.groupby(['model', 'diopter'])['qty'].sum().reset_index(name='qty_shipped')
    b_qty_elw = bill_elw.groupby(['model', 'diopter']).size().reset_index(name='qty_billed')
    
    inv_elw = pd.merge(s_qty_elw, b_qty_elw, on=['model', 'diopter'], how='outer').fillna(0)
    inv_elw['current_stock'] = inv_elw['qty_shipped'] - inv_elw['qty_billed']
    inv_elw['qty_supply'] = inv_elw['qty_shipped'] # For ELW, supply is what was shipped
    inv_elw['center_id'] = 'ELW'

    # 2. OCNH Inventory (Explicit from Stock File + Billing for Usage)
    # Default to calculated if file missing, but prioritize file
    inv_ocnh = pd.DataFrame()
    try:
        ocnh_stock_df = pd.read_csv('stock_OCNH_as_of_17-12-2025.csv')
        # Ensure columns match expectations
        ocnh_stock_df = ocnh_stock_df.rename(columns={'qty': 'current_stock'})
        
        # Get Usage from Billing
        bill_ocnh = billing_df[billing_df['center_id'] == 'OCNH']
        b_qty_ocnh = bill_ocnh.groupby(['model', 'diopter']).size().reset_index(name='qty_billed')
        
        # Merge File Stock with Billing Usage
        inv_ocnh = pd.merge(ocnh_stock_df, b_qty_ocnh, on=['model', 'diopter'], how='outer').fillna(0)
        inv_ocnh['center_id'] = 'OCNH'
        inv_ocnh['qty_shipped'] = 0 # Not using shipments for OCNH stock calc
        # For chart consistency, create a 'Supply' metric which is Stock + Usage
        inv_ocnh['qty_supply'] = inv_ocnh['current_stock'] + inv_ocnh['qty_billed']
        
    except FileNotFoundError:
        # Fallback to calculation if stock file missing
        ship_ocnh = shipments_df[shipments_df['center_id'] == 'OCNH']
        bill_ocnh = billing_df[billing_df['center_id'] == 'OCNH']
        s_qty_ocnh = ship_ocnh.groupby(['model', 'diopter'])['qty'].sum().reset_index(name='qty_shipped')
        b_qty_ocnh = bill_ocnh.groupby(['model', 'diopter']).size().reset_index(name='qty_billed')
        inv_ocnh = pd.merge(s_qty_ocnh, b_qty_ocnh, on=['model', 'diopter'], how='outer').fillna(0)
        inv_ocnh['current_stock'] = inv_ocnh['qty_shipped'] - inv_ocnh['qty_billed']
        inv_ocnh['qty_supply'] = inv_ocnh['qty_shipped']
        inv_ocnh['center_id'] = 'OCNH'

    # Combine Inventories
    inventory_df = pd.concat([inv_elw, inv_ocnh], ignore_index=True)

    # Dead Stock Logic
    inventory_df['status'] = 'Healthy'
    inventory_df.loc[(inventory_df['current_stock'] > 0) & (inventory_df['qty_billed'] == 0), 'status'] = 'Dead Stock'
    inventory_df.loc[(inventory_df['current_stock'] > inventory_df['qty_billed']) & (inventory_df['qty_billed'] > 0), 'status'] = 'Overstocked'
    
    # Calculate Stock Value
    inventory_df['unit_cogs'] = inventory_df['model'].map(lambda x: pricing.get(x, {}).get('COGS', 0))
    inventory_df['stock_value'] = inventory_df['current_stock'] * inventory_df['unit_cogs']

    return shipments_df, billing_df, shipment_events, inventory_df

# --- Load Data ---
shipments_df, billing_df, shipment_events, inventory_df = load_and_process_data()

if shipments_df is not None:
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("Configuration")
    view_mode = st.sidebar.radio("Select View:", ["Comparison (Side-by-Side)", "ELW Only", "OCNH Only"])
    
    # Filter Data based on selection
    if view_mode == "ELW Only":
        target_centers = ['ELW']
    elif view_mode == "OCNH Only":
        target_centers = ['OCNH']
    else:
        target_centers = ['ELW', 'OCNH']

    # Filter main dataframes
    f_bill = billing_df[billing_df['center_id'].isin(target_centers)]
    f_ship = shipment_events[shipment_events['center_id'].isin(target_centers)]
    f_inv = inventory_df[inventory_df['center_id'].isin(target_centers)]

    st.title(f"🏥 Surgery Center Performance Dashboard")
    
    # --- HEADER: PARAMETERS & ASSUMPTIONS ---
    # Calculate Date Range for Display
    min_date = shipments_df['order_date'].min().strftime('%d-%b-%Y')
    max_date = billing_df['use_date'].max().strftime('%d-%b-%Y')
    
    # Create the container
    with st.expander("ℹ️ Data Parameters & Assumptions", expanded=True):
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""
            **🗓️ Data Period** 
            \n{min_date} to {max_date}
            """)
        with p2:
            st.markdown("""
            **💶 Pricing Assumptions** 
            \nPC580: ASP €140 | COGS €50  
            PC545: ASP €90 | COGS €30
            """)
        with p3:
            st.markdown("""
            **🚚 Freight Costs** 
            \nSame Day: €120 | Next Day: €50 | Other: €25
            """)

    # --- HELPER FUNCTION FOR METRICS ---
    def calculate_center_metrics(cid):
        # Filter for specific center
        c_bill = billing_df[billing_df['center_id'] == cid]
        c_ship = shipment_events[shipment_events['center_id'] == cid]
        c_inv = inventory_df[inventory_df['center_id'] == cid]
        
        if c_bill.empty: return {}

        rev = c_bill['Revenue'].sum()
        cogs = c_bill['COGS'].sum()
        gm = c_bill['Gross_Margin'].sum()
        freight = c_ship['Freight_Cost'].sum()
        net = gm - freight
        margin_pct = (net/rev)*100 if rev > 0 else 0
        
        stock_qty = c_inv['current_stock'].sum()
        stock_val = c_inv['stock_value'].sum()
        
        # Months Supply
        months = 5 # approx based on data
        avg_monthly_usage = c_bill.shape[0] / months
        mos = stock_qty / avg_monthly_usage if avg_monthly_usage > 0 else 0
        
        return {
            'revenue': rev, 'net_profit': net, 'margin_pct': margin_pct,
            'stock_qty': stock_qty, 'stock_val': stock_val, 'mos': mos,
            'cogs': cogs, 'gm': gm, 'freight': freight
        }

    # --- KPI SECTION ---
    st.subheader("Key Performance Indicators")

    if view_mode == "Comparison (Side-by-Side)":
        # Calculate for both
        m_elw = calculate_center_metrics('ELW')
        m_ocnh = calculate_center_metrics('OCNH')

        # Create Comparison Layout
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown("##### Net Profit")
            st.metric("ELW", f"€{m_elw.get('net_profit',0):,.0f}", f"{m_elw.get('margin_pct',0):.1f}%")
            st.metric("OCNH", f"€{m_ocnh.get('net_profit',0):,.0f}", f"{m_ocnh.get('margin_pct',0):.1f}%")
        
        with c2:
            st.markdown("##### Revenue")
            st.metric("ELW", f"€{m_elw.get('revenue',0):,.0f}")
            st.metric("OCNH", f"€{m_ocnh.get('revenue',0):,.0f}")
            
        with c3:
            st.markdown("##### Stock (Units)")
            st.metric("ELW", f"{int(m_elw.get('stock_qty',0))}")
            st.metric("OCNH", f"{int(m_ocnh.get('stock_qty',0))}")

        with c4:
            st.markdown("##### Stock Value")
            st.metric("ELW", f"€{m_elw.get('stock_val',0):,.0f}")
            st.metric("OCNH", f"€{m_ocnh.get('stock_val',0):,.0f}")
            
        with c5:
            st.markdown("##### Months Supply")
            st.metric("ELW", f"{m_elw.get('mos',0):.1f} Mo")
            st.metric("OCNH", f"{m_ocnh.get('mos',0):.1f} Mo")
            
    else:
        # Single View Metrics
        m = calculate_center_metrics(target_centers[0])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Net Profit", f"€{m.get('net_profit',0):,.0f}", f"{m.get('margin_pct',0):.1f}% Margin")
        c2.metric("Revenue", f"€{m.get('revenue',0):,.0f}")
        c3.metric("Stock Units", f"{int(m.get('stock_qty',0))}")
        c4.metric("Stock Value", f"€{m.get('stock_val',0):,.0f}")
        c5.metric("Months Supply", f"{m.get('mos',0):.1f}")

    st.divider()

    # --- FINANCIAL DEEP DIVE (WATERFALLS) ---
    st.subheader("💰 Financial Structure Comparison")
    
    if view_mode == "Comparison (Side-by-Side)":
        col_w1, col_w2 = st.columns(2)
        
        metrics = [m_elw, m_ocnh]
        cols = [col_w1, col_w2]
        names = ['ELW', 'OCNH']
        
        for i in range(2):
            if not metrics[i]: continue
            with cols[i]:
                d = metrics[i]
                fig = go.Figure(go.Waterfall(
                    name = names[i], orientation = "v",
                    measure = ["relative", "relative", "total", "relative", "total"],
                    x = ["Revenue", "COGS", "Gross Margin", "Freight", "Net Profit"],
                    textposition = "outside",
                    text = [f"€{d['revenue']/1000:.1f}k", f"-€{d['cogs']/1000:.1f}k", f"€{d['gm']/1000:.1f}k", f"-€{d['freight']/1000:.1f}k", f"€{d['net_profit']/1000:.1f}k"],
                    y = [d['revenue'], -d['cogs'], d['gm'], -d['freight'], d['net_profit']],
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                ))
                fig.update_layout(title=f"{names[i]} Profitability Waterfall", height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Single Waterfall
        d = calculate_center_metrics(target_centers[0])
        fig = go.Figure(go.Waterfall(
            name = target_centers[0], orientation = "v",
            measure = ["relative", "relative", "total", "relative", "total"],
            x = ["Revenue", "COGS", "Gross Margin", "Freight", "Net Profit"],
            textposition = "outside",
            text = [f"€{d['revenue']/1000:.1f}k", f"-€{d['cogs']/1000:.1f}k", f"€{d['gm']/1000:.1f}k", f"-€{d['freight']/1000:.1f}k", f"€{d['net_profit']/1000:.1f}k"],
            y = [d['revenue'], -d['cogs'], d['gm'], -d['freight'], d['net_profit']],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title=f"Profitability Waterfall", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- FREIGHT ANALYSIS ---
    st.subheader("🚚 Logistics & Freight Efficiency")
    
    # Prepare Freight Data
    freight_agg = f_ship.groupby(['center_id', 'Freight_Cost'])['Freight_Cost'].agg(['sum', 'count']).reset_index()
    freight_agg.columns = ['center_id', 'Freight_Cost', 'Total_Spend', 'Count']
    
    def get_label(cost):
        if cost == 120: return "Same Day (Rush)"
        elif cost == 50: return "Next Day"
        else: return "Standard"
    freight_agg['Service_Level'] = freight_agg['Freight_Cost'].apply(get_label)

    if view_mode == "Comparison (Side-by-Side)":
        c_f1, c_f2 = st.columns(2)
        centers = ['ELW', 'OCNH']
        cols = [c_f1, c_f2]
        
        for i in range(2):
            with cols[i]:
                data = freight_agg[freight_agg['center_id'] == centers[i]]
                if not data.empty:
                    fig_pie = px.pie(data, values='Total_Spend', names='Service_Level', 
                                     title=f'{centers[i]} Freight Spend', 
                                     hole=0.4,
                                     color='Service_Level',
                                     color_discrete_map={'Same Day (Rush)': '#EF553B', 'Next Day': '#FFA15A', 'Standard': '#00CC96'})
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
         fig_pie = px.pie(freight_agg, values='Total_Spend', names='Service_Level', 
                         title='Freight Spend Breakdown', 
                         hole=0.4,
                         color='Service_Level',
                         color_discrete_map={'Same Day (Rush)': '#EF553B', 'Next Day': '#FFA15A', 'Standard': '#00CC96'})
         st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # --- INVENTORY HEALTH ---
    st.subheader("📦 Inventory Balance (Supply vs Demand)")
    
    c_inv1, c_inv2 = st.columns([2, 1])
    
    with c_inv1:
        st.markdown("##### Supply vs Demand Analysis")
        # Interactive Filters
        c_filt1, c_filt2 = st.columns(2)
        with c_filt1:
            # Dropdown for Center
            avail_centers = f_inv['center_id'].unique()
            sel_center = st.selectbox("Select Center", avail_centers)
        
        with c_filt2:
            # Dropdown for Model (Dynamic based on center)
            avail_models = f_inv[f_inv['center_id'] == sel_center]['model'].unique()
            sel_model = st.selectbox("Select Model", avail_models)
            
        # Filter Data for Chart
        chart_data = f_inv[(f_inv['center_id'] == sel_center) & (f_inv['model'] == sel_model)].copy()
        # Filter to remove empty noise
        chart_data = chart_data[(chart_data['qty_supply'] > 0) | (chart_data['qty_billed'] > 0)]
        
        if not chart_data.empty:
            fig_bar = go.Figure()
            # 1. Total Supply (Shipped/Available) - Grey
            fig_bar.add_trace(go.Bar(
                x=chart_data['diopter'], 
                y=chart_data['qty_supply'], # Uses 'qty_supply' to ensure it works for both ELW (Shipped) and OCNH (Total Supply)
                name='Total Shipped/Available', 
                marker_color='lightgrey'
            ))
            # 2. Used - Green
            fig_bar.add_trace(go.Bar(
                x=chart_data['diopter'], 
                y=chart_data['qty_billed'], 
                name='Total Used', 
                marker_color='#00CC96'
            ))
            
            fig_bar.update_layout(
                title=f"Supply vs. Demand by Diopter: {sel_model} ({sel_center})",
                xaxis_title="Diopter Power",
                yaxis_title="Quantity",
                barmode='overlay', # Overlay mode as requested
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("Insight: Grey bars visible above green bars indicate excess inventory sitting on the shelf.")
        else:
            st.warning("No data available for this selection.")

    with c_inv2:
        # Status Check
        status_counts = f_inv.groupby(['center_id', 'status']).size().reset_index(name='Count')
        fig_status = px.bar(status_counts, x='center_id', y='Count', color='status', 
                            title="SKU Health Status",
                            color_discrete_map={'Healthy': '#00CC96', 'Overstocked': '#FFA15A', 'Dead Stock': '#EF553B'})
        st.plotly_chart(fig_status, use_container_width=True)

    # --- ACTION ITEMS ---
    st.subheader("📋 Action Items")
    
    tab1, tab2 = st.tabs(["🔴 Dead Stock Candidates", "🟡 Overstock Warnings"])
    
    with tab1:
        dead_stock = f_inv[f_inv['status'] == 'Dead Stock'].sort_values(['center_id', 'current_stock'], ascending=[True, False])
        st.dataframe(dead_stock[['center_id', 'model', 'diopter', 'current_stock', 'stock_value']], use_container_width=True)
    
    with tab2:
        overstock = f_inv[(f_inv['status'] == 'Overstocked') & (f_inv['current_stock'] > 5)].sort_values(['center_id', 'current_stock'], ascending=[True, False])
        st.dataframe(overstock[['center_id', 'model', 'diopter', 'qty_shipped', 'qty_billed', 'current_stock']], use_container_width=True)

    st.divider()

    # --- MONTH OVER MONTH COMPARISON ---
    st.subheader("📈 Month Over Month Comparison by Week")
    
    # --- 1. Data Preparation (Combined Weeks Logic) ---
    def process_weekly_data(df, date_col, is_shipment=False):
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col])
        iso = temp[date_col].dt.isocalendar()
        temp['iso_year'] = iso.year
        temp['iso_week'] = iso.week
        temp['month_name'] = temp[date_col].dt.strftime('%b')
        
        # Grouping Keys
        grp_cols = ['center_id', 'iso_year', 'iso_week']
        
        if is_shipment:
            return temp.groupby(grp_cols).agg({
                'Freight_Cost': 'sum',
                'month_name': lambda x: set(x)
            }).reset_index()
        else:
            return temp.groupby(grp_cols).agg({
                'Revenue': 'sum', 
                'COGS': 'sum', 
                'Gross_Margin': 'sum',
                'model': 'count',
                'month_name': lambda x: set(x)
            }).rename(columns={'model': 'Units_Used'}).reset_index()

    # --- SECTION FILTERS ---
    c_filt_main1, c_filt_main2 = st.columns(2)
    
    with c_filt_main1:
        avail_centers_all = billing_df['center_id'].unique()
        sel_centers_mom = st.multiselect("Select Surgery Centers", options=avail_centers_all, default=avail_centers_all)

    # Process Data
    b_raw = billing_df[billing_df['center_id'].isin(sel_centers_mom)]
    s_raw = shipment_events[shipment_events['center_id'].isin(sel_centers_mom)]

    b_grouped = process_weekly_data(b_raw, 'use_date', is_shipment=False)
    s_grouped = process_weekly_data(s_raw, 'delivery_date', is_shipment=True)

    weekly_df = pd.merge(b_grouped, s_grouped, on=['center_id', 'iso_year', 'iso_week'], how='outer')
    
    weekly_df['Freight_Cost'] = weekly_df['Freight_Cost'].fillna(0)
    weekly_df[['Revenue', 'COGS', 'Gross_Margin', 'Units_Used']] = weekly_df[['Revenue', 'COGS', 'Gross_Margin', 'Units_Used']].fillna(0)
    
    # === NEW LOGIC: CREATE UNIFIED GLOBAL LABELS ===
    # 1. Combine month sets from both columns
    def combine_row_months(row):
        m1 = row['month_name_x'] if isinstance(row['month_name_x'], set) else set()
        m2 = row['month_name_y'] if isinstance(row['month_name_y'], set) else set()
        return m1.union(m2)
    
    weekly_df['center_months'] = weekly_df.apply(combine_row_months, axis=1)

    # 2. Create a "Master Dictionary" for Year-Week -> All Months involved
    # This ensures that if ANY center touched Nov in Week 44, EVERYONE calls it "Oct/Nov"
    global_week_map = {}
    
    # Group by Year-Week and combine all sets of months found
    for (year, week), group in weekly_df.groupby(['iso_year', 'iso_week']):
        all_months_in_week = set()
        for month_set in group['center_months']:
            all_months_in_week.update(month_set)
        global_week_map[(year, week)] = all_months_in_week

    month_order = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 
                   'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

    # 3. Generate the Label based on the GLOBAL map, not the row data
    def get_global_label(row):
        year, week = row['iso_year'], row['iso_week']
        months_set = global_week_map.get((year, week), set())
        
        sorted_months = sorted(list(months_set), key=lambda x: month_order.get(x, 0))
        label_months = "/".join(sorted_months)
        return f"{label_months} Wk {week}"

    weekly_df['period_label'] = weekly_df.apply(get_global_label, axis=1)
    weekly_df['Net_Profit'] = weekly_df['Gross_Margin'] - weekly_df['Freight_Cost']
    
    # Sort chronologically
    weekly_df = weekly_df.sort_values(['iso_year', 'iso_week'])

    # --- FILTER UI ---
    with c_filt_main2:
        # Determine unique months from the GLOBAL map for the dropdown
        all_present_months = set()
        for m_set in global_week_map.values():
            all_present_months.update(m_set)
            
        sorted_month_opts = sorted(list(all_present_months), key=lambda x: month_order.get(x, 0))
        sel_months = st.multiselect("Select Months", options=sorted_month_opts, default=sorted_month_opts)

    # Filter logic uses the row's specific months (to allow filtering), 
    # but the LABEL remains consistent.
    def filter_by_month(row_months):
        return not row_months.isdisjoint(set(sel_months))

    chart_df = weekly_df[weekly_df['center_months'].apply(filter_by_month)].copy()

    if not chart_df.empty:
        # Define categorical order
        week_order = list(dict.fromkeys(chart_df['period_label']))

        col_g1, col_g2 = st.columns(2)
        
        # --- GRAPH 1: UNITS USED ---
        with col_g1:
            st.markdown("##### 📦 Units Used Over Week")
            fig_units = px.line(chart_df, x='period_label', y='Units_Used', color='center_id',
                                markers=True,
                                labels={'period_label': 'Week', 'Units_Used': 'Units', 'center_id': 'Center'},
                                color_discrete_map={'ELW': '#0068C9', 'OCNH': '#83C9FF'})
            
            fig_units.update_xaxes(categoryorder='array', categoryarray=week_order)
            fig_units.update_layout(xaxis_title=None, height=400, hovermode="x unified")
            st.plotly_chart(fig_units, use_container_width=True)
            
        # --- GRAPH 2: FINANCIALS ---
        with col_g2:
            # 1. Header Row: Title (Left) + Toggle (Right)
            fh_1, fh_2 = st.columns([0.65, 0.35])
            with fh_1:
                st.markdown("##### 💰 Financials Over Week")
            with fh_2:
                # Align toggle slightly down to match header text baseline if needed
                split_center = st.toggle("Split by Center", value=False)
            
            # 2. Content Row: Chart (Left) + Checkboxes (Right)
            fc_chart, fc_opts = st.columns([0.75, 0.25])
            
            # -- Right Column: Checkboxes --
            with fc_opts:
                st.write("") # Small spacer
                st.write("") 
                
                # Mapping Display Labels -> Dataframe Columns
                metrics_map = {
                    "Revenue": "Revenue",
                    "Cost of Goods Sold": "COGS", 
                    "Gross Margin": "Gross_Margin",
                    "Freight Charges": "Freight_Cost", 
                    "Net Profit": "Net_Profit"
                }
                
                sel_metrics = []
                for label, col_name in metrics_map.items():
                    # Set default checked items
                    is_checked = col_name in ['Revenue', 'Net_Profit']
                    if st.checkbox(label, value=is_checked):
                        sel_metrics.append(col_name)

            # -- Left Column: Chart --
            with fc_chart:
                if sel_metrics:
                    if split_center:
                        melted = chart_df.melt(id_vars=['period_label', 'center_id'], value_vars=sel_metrics, var_name='Metric', value_name='Value')
                        fig_fin = px.line(melted, x='period_label', y='Value', color='Metric', line_dash='center_id', markers=True)
                    else:
                        agg_df = chart_df.groupby('period_label', sort=False)[sel_metrics].sum().reset_index()
                        melted = agg_df.melt(id_vars=['period_label'], value_vars=sel_metrics, var_name='Metric', value_name='Value')
                        fig_fin = px.line(melted, x='period_label', y='Value', color='Metric', markers=True)
                    
                    fig_fin.update_xaxes(categoryorder='array', categoryarray=week_order)
                    
                    # Moved legend to top to avoid cluttering the visual space
                    fig_fin.update_layout(
                        xaxis_title=None, 
                        yaxis_title="Amount (€)", 
                        height=400, 
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=10, b=0), # Tighten margins
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_fin, use_container_width=True)
                else:
                    st.warning("Select metrics.")
                    
    st.divider()

    # --- INVENTORY DYNAMICS SECTION ---
    st.subheader("📦 Inventory Dynamics Over Week")

    # 1. Prepare Data: Get Unit Quantities AND Values (Lines)
    # We need raw 'qty' from shipments (s_raw only had Freight Cost aggregated)
    s_raw_lines = shipments_df[shipments_df['center_id'].isin(sel_centers_mom)].copy()
    s_raw_lines['delivery_date'] = pd.to_datetime(s_raw_lines['delivery_date'])
    
    # --- Calculate Value of Inbound Shipments (Exact COGS) ---
    cogs_map = {'PC580': 50, 'PC545': 30}
    s_raw_lines['unit_cogs'] = s_raw_lines['model'].map(cogs_map).fillna(0)
    s_raw_lines['shipment_value'] = s_raw_lines['qty'] * s_raw_lines['unit_cogs']

    # Add ISO Date Info to Shipment Lines
    iso_s = s_raw_lines['delivery_date'].dt.isocalendar()
    s_raw_lines['iso_year'] = iso_s.year
    s_raw_lines['iso_week'] = iso_s.week
    
    # Group to get Total Units AND Value Shipped per Week
    s_qty_grouped = s_raw_lines.groupby(['center_id', 'iso_year', 'iso_week']).agg({
        'qty': 'sum',
        'shipment_value': 'sum'
    }).reset_index().rename(columns={'qty': 'Units_Shipped', 'shipment_value': 'Value_Shipped'})

    # Merge Units Shipped into the Main Weekly Timeline
    # (weekly_df already contains Units_Used and COGS from Billing)
    inv_trend_df = pd.merge(weekly_df, s_qty_grouped, on=['center_id', 'iso_year', 'iso_week'], how='left')
    inv_trend_df['Units_Shipped'] = inv_trend_df['Units_Shipped'].fillna(0)
    inv_trend_df['Value_Shipped'] = inv_trend_df['Value_Shipped'].fillna(0)
    
    # 2. Calculate Inventory Trends (Running Totals)
    # Quantity Flow
    inv_trend_df['Net_Change_Qty'] = inv_trend_df['Units_Shipped'] - inv_trend_df['Units_Used']
    
    # Value Flow (Inbound Value - Outbound COGS)
    inv_trend_df['Net_Change_Value'] = inv_trend_df['Value_Shipped'] - inv_trend_df['COGS']

    # Ensure correct sort order for Cumulative Sum
    inv_trend_df = inv_trend_df.sort_values(['center_id', 'iso_year', 'iso_week'])
    
    # Cumulative Sum: This reconstructs the historical stock level
    inv_trend_df['Closing_Stock_Qty'] = inv_trend_df.groupby('center_id')['Net_Change_Qty'].cumsum()
    
    # Calculate Exact Value on Hand
    inv_trend_df['Closing_Stock_Value'] = inv_trend_df.groupby('center_id')['Net_Change_Value'].cumsum()
    
    # Months of Supply (MoS) Trend
    # We use a Rolling 4-Week Average of Usage to smooth out volatility
    inv_trend_df['Rolling_Usage'] = inv_trend_df.groupby('center_id')['Units_Used'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    
    # Calculate MoS: Stock / (Weekly Usage * 4.3 weeks/month)
    inv_trend_df['MoS_Trend'] = inv_trend_df.apply(
        lambda x: x['Closing_Stock_Qty'] / (x['Rolling_Usage'] * 4.3) if x['Rolling_Usage'] > 0 else 0, 
        axis=1
    )

    # 3. Apply Filter for Display (Syncs with the Month Selector)
    inv_chart_df = inv_trend_df[inv_trend_df['center_months'].apply(filter_by_month)].copy()

    if not inv_chart_df.empty:
        # Re-establish week order for this chart in case filter changed
        inv_week_order = list(dict.fromkeys(inv_chart_df['period_label']))
        
        col_i1, col_i2 = st.columns(2)
        
        # --- CHART 1: FLOW (IN vs OUT) ---
        with col_i1:
            st.markdown("##### 🔄 Net Flow (Shipped vs. Used)")
            # Melt data for Grouped Bar Chart
            flow_df = inv_chart_df.melt(id_vars=['period_label', 'center_id'], 
                                        value_vars=['Units_Shipped', 'Units_Used'],
                                        var_name='Type', value_name='Qty')
            
            # Rename for legend clarity
            flow_df['Type'] = flow_df['Type'].map({'Units_Shipped': 'Restock (In)', 'Units_Used': 'Consumption (Out)'})
            
            fig_flow = px.bar(flow_df, x='period_label', y='Qty', color='Type',
                                barmode='group',
                                facet_row='center_id', # Split centers to keep it readable
                                color_discrete_map={'Restock (In)': '#0068C9', 'Consumption (Out)': '#00CC96'},
                                labels={'period_label': 'Week', 'Qty': 'Units'},
                                height=500)
            
            fig_flow.update_xaxes(title=None, categoryorder='array', categoryarray=inv_week_order)
            fig_flow.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_flow, use_container_width=True)
            
        # --- CHART 2: STOCK HOLDING TREND ---
        with col_i2:
            st.markdown("##### 📈 Inventory Holding Trend")
            
            # Toggle for Metric Selection
            view_opt = st.radio("Select View:", ["Units On-Hand", "Value On-Hand (€)", "Months of Supply"], horizontal=True, label_visibility="collapsed")
            
            # Map selection to DataFrame column
            y_map = {
                "Units On-Hand": "Closing_Stock_Qty", 
                "Value On-Hand (€)": "Closing_Stock_Value", 
                "Months of Supply": "MoS_Trend"
            }
            
            fig_trend = px.line(inv_chart_df, x='period_label', y=y_map[view_opt], color='center_id',
                                markers=True,
                                color_discrete_map={'ELW': '#EF553B', 'OCNH': '#FFA15A'},
                                height=450)
            
            fig_trend.update_xaxes(title=None, categoryorder='array', categoryarray=inv_week_order)
            fig_trend.update_yaxes(rangemode="tozero") # Anchor y-axis to 0
            fig_trend.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            #if view_opt == "Months of Supply":
                #st.caption("ℹ️ **Insight:** A rising MoS trend (while sales are flat) is an early warning signal of dead stock accumulation.")

    else:
        st.info("No data available for the selected range.")

else:
    st.info("Loading interface... if data is missing, this screen will persist.")