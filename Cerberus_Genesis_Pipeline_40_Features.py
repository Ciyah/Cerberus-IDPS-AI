import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

print("=======================================================")
print("CERBERUS PHASE 1 & 2: HETEROGENEOUS MODEL SYNTHESIS ")
print("Engine: HGBM (40-Feature Extreme Physics Matrix)")
print("=======================================================\n")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point these to exact dataset  CSV files
DATASET_A_PATH = os.path.join(BASE_DIR, "CICIoT2023", "CICIOT23", "train", "train.csv")
DATASET_B_PATH = os.path.join(BASE_DIR, "IoTID20", "IoT Network Intrusion Dataset.csv")

# ==========================================
# CORE ENGINEERING FUNCTIONS
# ==========================================
def extract_extreme_physics_schema(df, dataset_type):
    """
    The 40-Feature Extreme Physics Extraction Layer.
    Maximized for Deep HGBM Analysis without crashing Edge Hardware.
    """
    df_clean = pd.DataFrame()
    
    if dataset_type == "CIC":
        # 1. Core Flow Mathematics
        df_clean['flow_duration'] = df.get('Duration', df.get('flow_duration', 0))
        df_clean['flow_rate'] = df.get('Rate', 0)
        df_clean['pkt_len_mean'] = df.get('AVG', 0)
        df_clean['pkt_len_std'] = df.get('Std', 0)
        df_clean['pkt_len_max'] = df.get('Max', 0)
        df_clean['pkt_len_min'] = df.get('Min', 0)
        df_clean['pkt_len_var'] = df.get('Variance', 0)
        df_clean['flow_iat_mean'] = df.get('IAT', 0)
        df_clean['header_length'] = df.get('Header_Length', 0)
        
        # 2. Directional Flow (Asymmetry)
        df_clean['fwd_rate'] = df.get('Srate', 0)
        df_clean['bwd_rate'] = df.get('Drate', 0)
        df_clean['fwd_pkts_tot'] = df.get('Number', 0) 
        df_clean['bwd_pkts_tot'] = 0 
        df_clean['fwd_header_len'] = df.get('Header_Length', 0) / 2
        df_clean['bwd_header_len'] = df.get('Header_Length', 0) / 2
        
        # 3. Expanded TCP Flags
        df_clean['syn_flag'] = df.get('syn_flag_number', 0)
        df_clean['ack_flag'] = df.get('ack_flag_number', 0)
        df_clean['rst_flag'] = df.get('rst_flag_number', 0)
        df_clean['fin_flag'] = df.get('fin_flag_number', 0)
        df_clean['psh_flag'] = df.get('psh_flag_number', 0)
        df_clean['urg_flag'] = df.get('urg_flag_number', 0)
        df_clean['cwe_flag'] = df.get('cwr_flag_number', 0) 
        
        # 4. Active / Idle Beaconing
        df_clean['active_mean'] = df.get('Active_Mean', 0)
        df_clean['active_std'] = df.get('Active_Std', 0)
        df_clean['idle_mean'] = df.get('Idle_Mean', 0)
        df_clean['idle_std'] = df.get('Idle_Std', 0)
        
        # 5. Core Protocols (L2/L3/L4)
        df_clean['is_tcp'] = df.get('TCP', 0)
        df_clean['is_udp'] = df.get('UDP', 0)
        df_clean['is_icmp'] = df.get('ICMP', 0)
        df_clean['is_arp'] = df.get('ARP', 0)
        df_clean['is_ipv4'] = df.get('IPv', 1) 
        
        # 6. Expanded Application Layer (L7) Port Mapping
        df_clean['is_http'] = df.get('HTTP', 0)
        df_clean['is_https'] = df.get('HTTPS', 0)
        df_clean['is_dns'] = df.get('DNS', 0)
        df_clean['is_ssh'] = df.get('SSH', 0)
        df_clean['is_telnet'] = df.get('Telnet', 0)
        df_clean['is_dhcp'] = df.get('DHCP', 0)
        df_clean['is_smtp'] = df.get('SMTP', 0)
        df_clean['is_irc'] = df.get('IRC', 0)
        
    elif dataset_type == "IoTID":
        # 1. Core Flow Mathematics
        df_clean['flow_duration'] = df.get('Flow_Duration', 0)
        df_clean['flow_rate'] = df.get('Flow_Pkts/s', 0)
        df_clean['pkt_len_mean'] = df.get('Pkt_Len_Mean', 0)
        df_clean['pkt_len_std'] = df.get('Pkt_Len_Std', 0)
        df_clean['pkt_len_max'] = df.get('Pkt_Len_Max', 0)
        df_clean['pkt_len_min'] = df.get('Pkt_Len_Min', 0)
        df_clean['pkt_len_var'] = df.get('Pkt_Len_Var', 0)
        df_clean['flow_iat_mean'] = df.get('Flow_IAT_Mean', 0)
        df_clean['header_length'] = df.get('Fwd_Header_Length', 0) + df.get('Bwd_Header_Length', 0)
        
        # 2. Directional Flow (Asymmetry)
        df_clean['fwd_rate'] = df.get('Fwd_Pkts/s', 0)
        df_clean['bwd_rate'] = df.get('Bwd_Pkts/s', 0)
        df_clean['fwd_pkts_tot'] = df.get('Tot_Fwd_Pkts', 0)
        df_clean['bwd_pkts_tot'] = df.get('Tot_Bwd_Pkts', 0)
        df_clean['fwd_header_len'] = df.get('Fwd_Header_Length', 0)
        df_clean['bwd_header_len'] = df.get('Bwd_Header_Length', 0)
        
        # 3. Expanded TCP Flags
        df_clean['syn_flag'] = df.get('SYN_Flag_Cnt', 0)
        df_clean['ack_flag'] = df.get('ACK_Flag_Cnt', 0)
        df_clean['rst_flag'] = df.get('RST_Flag_Cnt', 0)
        df_clean['fin_flag'] = df.get('FIN_Flag_Cnt', 0)
        df_clean['psh_flag'] = df.get('PSH_Flag_Cnt', 0)
        df_clean['urg_flag'] = df.get('URG_Flag_Cnt', 0)
        df_clean['cwe_flag'] = df.get('CWE_Flag_Count', 0)
        df_clean['ece_flag'] = df.get('ECE_Flag_Cnt', 0)
        
        # 4. Active / Idle Beaconing
        df_clean['active_mean'] = df.get('Active_Mean', 0)
        df_clean['active_std'] = df.get('Active_Std', 0)
        df_clean['idle_mean'] = df.get('Idle_Mean', 0)
        df_clean['idle_std'] = df.get('Idle_Std', 0)
        
        # 5. Core Protocols & Tie-Breakers
        protocol = df.get('Protocol', 17)
        df_clean['is_tcp'] = (protocol == 6).astype(int)
        df_clean['is_udp'] = (protocol == 17).astype(int)
        df_clean['is_icmp'] = (protocol == 1).astype(int)
        df_clean['is_arp'] = ((protocol != 6) & (protocol != 17) & (protocol != 1)).astype(int)
        df_clean['is_ipv4'] = 1 
        
        # 6. Mathematical Application Layer (L7) Derivation from Ports
        dst_port = df.get('Dst_Port', 0)
        df_clean['is_http'] = (dst_port == 80).astype(int)
        df_clean['is_https'] = (dst_port == 443).astype(int)
        df_clean['is_dns'] = (dst_port == 53).astype(int)
        df_clean['is_ssh'] = (dst_port == 22).astype(int)
        df_clean['is_telnet'] = (dst_port == 23).astype(int)
        df_clean['is_dhcp'] = ((dst_port == 67) | (dst_port == 68)).astype(int)
        df_clean['is_smtp'] = (dst_port == 25).astype(int)
        df_clean['is_irc'] = (dst_port == 6667).astype(int)

    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.fillna(0, inplace=True)
    return df_clean

def map_universal_threats(lbl_value):
    lbl = str(lbl_value).strip().lower()
    if lbl in ['benigntraffic', '0', 'benign', 'normal']: return 0
    elif any(kw in lbl for kw in ['dos', 'ddos', 'flood', 'syn', 'mirai']): return 2
    elif any(kw in lbl for kw in ['mitm', 'arp', 'spoof', 'dns']): return 1
    return -1

def apply_local_baseline(df):
    """
    ADDED: Local Baseline Normalization (LBN). Z-scores each source
    dataset's continuous features BEFORE merging, independently per source.
    This matters specifically because CICIoT2023 and IoTID20 measure the
    same physical quantities (duration, packet size, etc.) in different
    absolute ranges/units -- without this, a single set of tree splits
    can't cleanly separate patterns across both sources at once.
    """
    scaler = StandardScaler()
    binary_cols = ['is_tcp', 'is_udp', 'is_icmp', 'is_arp', 'is_ipv4',
                   'is_http', 'is_https', 'is_dns', 'is_ssh', 'is_telnet',
                   'is_dhcp', 'is_smtp', 'is_irc', 'target_label']
    cont_cols = [c for c in df.columns if c not in binary_cols]
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    return df

# ==========================================
# 1. HETEROGENEOUS INGESTION & DATA LOGGING
# ==========================================
print("[*] Ingesting Dataset A (CICIoT2023)...")
try:
    df_cic_raw = pd.read_csv(DATASET_A_PATH) 
    df_cic_eval = extract_extreme_physics_schema(df_cic_raw, "CIC")
    df_cic_eval['target_label'] = df_cic_raw['label'].apply(map_universal_threats) 
    df_cic_eval = apply_local_baseline(df_cic_eval)
    count_a = len(df_cic_eval)
    print(f"    -> Successfully extracted {count_a:,} total rows.")
    
    counts_a = df_cic_eval['target_label'].value_counts()
    print(f"       Class 0 [Normal]: {counts_a.get(0, 0):,} rows")
    print(f"       Class 1 [MitM]:   {counts_a.get(1, 0):,} rows")
    print(f"       Class 2 [DoS]:    {counts_a.get(2, 0):,} rows")
except Exception as e:
    print(f"[!] Error loading CICIoT2023: {e}")
    exit()

print("\n[*] Ingesting Dataset B (IoTID20)...")
try:
    df_iot_raw = pd.read_csv(DATASET_B_PATH)
    df_iot_eval = extract_extreme_physics_schema(df_iot_raw, "IoTID")
    label_col = 'Cat' if 'Cat' in df_iot_raw.columns else 'Label' 
    df_iot_eval['target_label'] = df_iot_raw[label_col].apply(map_universal_threats)
    df_iot_eval = apply_local_baseline(df_iot_eval)
    count_b = len(df_iot_eval)
    print(f"    -> Successfully extracted {count_b:,} total rows.")
    
    counts_b = df_iot_eval['target_label'].value_counts()
    print(f"       Class 0 [Normal]: {counts_b.get(0, 0):,} rows")
    print(f"       Class 1 [MitM]:   {counts_b.get(1, 0):,} rows")
    print(f"       Class 2 [DoS]:    {counts_b.get(2, 0):,} rows")
except Exception as e:
    print(f"[!] Error loading IoTID20: {e}")
    exit()

print("\n[*] Executing Heterogeneous Merge...")
print(f"    -> Dataset A Pre-Merge: {count_a:,} rows")
print(f"    -> Dataset B Pre-Merge: {count_b:,} rows")

df_combined = pd.concat([df_cic_eval, df_iot_eval], ignore_index=True)
raw_combined = len(df_combined)
print(f"    -> Total Raw Merged:    {raw_combined:,} rows")

df_combined = df_combined[df_combined['target_label'] != -1]
final_combined = len(df_combined)
dropped = raw_combined - final_combined
print(f"    -> Unmapped Threats Dropped: {dropped:,} rows")
print(f"    -> FINAL Valid Dataset:      {final_combined:,} rows\n")

# ==========================================
# 2. ADVANCED CLEANING & STRATIFICATION
# ==========================================
print("[*] Executing Advanced Data Cleaning (Toxic Vector Purge)...")
df_combined = df_combined.drop_duplicates()

feature_cols = [c for c in df_combined.columns if c != 'target_label']
pre_purge_count = len(df_combined)

pby(feature_cols)['target_label'].transform('nunique')
df_combined = df_combined[conflict_counts == 1]
df_combined = df_combined.drop_duplicates(subset=feature_cols, keep='first')

purged_count = pre_purge_count - len(df_combined)
print(f"    -> Purged {purged_count:,} genuinely toxic/ambiguous or duplicate rows.\n")

df_norm = df_combined[df_combined['target_label'] == 0]
df_mitm = df_combined[df_combined['target_label'] == 1]
df_dos  = df_combined[df_combined['target_label'] == 2]

true_cap = min(len(df_norm), len(df_mitm), len(df_dos), 100000)

print(f"[*] Locking exact equilibrium matrix at {true_cap:,} unique rows per class...")

df_massive = pd.concat([
    df_norm.sample(n=true_cap, random_state=42),
    df_mitm.sample(n=true_cap, random_state=42),
    df_dos.sample(n=true_cap, random_state=42)
]).sample(frac=1, random_state=42)

X = df_massive.drop(columns=['target_label'])
y = df_massive['target_label']

# --- FLOWCHART ALIGNMENT: 80% Train, 20% Test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# ==========================================
# 3. MODERN HISTOGRAM GRADIENT BOOSTING
# ==========================================
print(f"[*] Training Cerberus HGBM on {len(X.columns)} EXTREME features...")

model = HistGradientBoostingClassifier(
    # Retuned to the settings validated earlier on this same architecture
    # family: lower learning_rate + more max_iter carves finer decision
    # boundaries instead of converging early on the "easy" DoS split and
    # leaving Normal/MitM under-refined; max_leaf_nodes/min_samples_leaf
    # give the model room to do that without overfitting; more
    # n_iter_no_change patience stops it from quitting right after DoS
    # converges but before Normal/MitM has finished improving.
    max_iter=1200,
    learning_rate=0.04,
    max_leaf_nodes=255,
    min_samples_leaf=15,
    l2_regularization=0.05,
    max_bins=255,
    max_depth=None,
    class_weight='balanced', 
    random_state=42,
    
    # --- FLOWCHART ALIGNMENT: The 70/10 Split (unchanged) ---
    # early_stopping=True tells the algorithm to evaluate itself.
    # validation_fraction=0.125 tells it to carve out 12.5% of the X_train data.
    # (12.5% of 80% = Exactly 10% of total data for Validation).
    # (Leaving exactly 70% of total data purely for Training).
    early_stopping=True,
    validation_fraction=0.125,
    n_iter_no_change=30
)
model.fit(X_train, y_train)

# ==========================================
# 4. EXPORT & VALIDATE
# ==========================================
print("\n=======================================================")
print("SCIENTIFIC VALIDATION SCORE (40-FEATURE HGBM)")
print("=======================================================")

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
# labels=[0,1,2] is explicit so all three classes always print, in the
# correct order, even if a class happens to be thin in a given split.
print(classification_report(
    y_test, y_pred,
    labels=[0, 1, 2],
    target_names=['Normal (0)', 'MitM (1)', 'Volumetric DoS (2)'],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print("Confusion Matrix (rows=true, cols=pred) [Normal, MitM, DoS]:")
print(cm, "\n")

joblib.dump(model, os.path.join(BASE_DIR, 'Cerberus_HGBM_Brain.joblib'))
print("[+] Export Complete. Ready for Phase 3: Native Architecture Transfer\n")
