import pandas as pd
import numpy as np
import os
import glob
import warnings
import time
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIAD_DATA_DIR = os.path.join(BASE_DIR, "CIC-DIAD")
MODEL_PATH = os.path.join(BASE_DIR, "Cerberus_HGBM_Brain.joblib")
CLASS_ROW_BUDGET = {0: 150_000, 1: 150_000, 2: 150_000}
FINETUNE_FRACTION = 0.50
FINETUNE_EXTRA_ITERS = 600


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


def extract_extreme_physics_schema(df):
    df_clean = pd.DataFrame(index=df.index)
    cols = {str(c).lower().replace(' ', '').replace('_', ''): c for c in df.columns}

    def grab(*names):
        for n in names:
            clean_n = n.lower().replace(' ', '').replace('_', '')
            if clean_n in cols: return df[cols[clean_n]]
        return 0

    df_clean['flow_duration'] = grab('flowduration', 'duration')
    df_clean['flow_rate'] = grab('flowpackets/s', 'flowrate', 'flowbytes/s')
    df_clean['pkt_len_mean'] = grab('pktlenmean', 'packetlengthmean', 'meanpacketlength')
    df_clean['pkt_len_std'] = grab('pktlenstd', 'packetlengthstd', 'stdpacketlength')
    df_clean['pkt_len_max'] = grab('pktlenmax', 'packetlengthmax', 'maxpacketlength')
    df_clean['pkt_len_min'] = grab('pktlenmin', 'packetlengthmin', 'minpacketlength')
    df_clean['pkt_len_var'] = grab('pktlenvar', 'packetlengthvariance', 'variancepacketlength')
    df_clean['flow_iat_mean'] = grab('flowiatmean', 'iatmean')
    df_clean['header_length'] = grab('fwdheaderlength') + grab('bwdheaderlength')

    df_clean['fwd_rate'] = grab('fwdpackets/s', 'srate', 'fwdpkts/s')
    df_clean['bwd_rate'] = grab('bwdpackets/s', 'drate', 'bwdpkts/s')
    df_clean['fwd_pkts_tot'] = grab('totalfwdpacket', 'totfwdpkts', 'totalforwardpackets')
    df_clean['bwd_pkts_tot'] = grab('totalbwdpackets', 'totbwdpkts')
    df_clean['fwd_header_len'] = grab('fwdheaderlength')
    df_clean['bwd_header_len'] = grab('bwdheaderlength')

    df_clean['syn_flag'] = grab('syncount', 'synflagnumber', 'synflagcnt', 'synflagcount')
    df_clean['ack_flag'] = grab('ackcount', 'ackflagnumber', 'ackflagcnt', 'ackflagcount')
    df_clean['rst_flag'] = grab('rstcount', 'rstflagnumber', 'rstflagcnt', 'rstflagcount')
    df_clean['fin_flag'] = grab('fincount', 'finflagnumber', 'finflagcnt', 'finflagcount')
    df_clean['psh_flag'] = grab('pshflagnumber', 'pshflagcnt', 'pshflagcount')
    df_clean['urg_flag'] = grab('urgcount', 'urgflagnumber', 'urgflagcnt', 'urgflagcount')
    df_clean['cwe_flag'] = grab('cwrflagcount', 'cweflagcount', 'cweflagcnt', 'cwrflagnumber')
    df_clean['ece_flag'] = grab('eceflagnumber', 'eceflagcnt', 'eceflagcount')

    df_clean['active_mean'] = grab('activemean')
    df_clean['active_std'] = grab('activestd')
    df_clean['idle_mean'] = grab('idlemean')
    df_clean['idle_std'] = grab('idlestd')

    df_clean['is_tcp'] = grab('tcp')
    df_clean['is_udp'] = grab('udp')
    df_clean['is_icmp'] = grab('icmp')
    df_clean['is_arp'] = grab('arp')
    df_clean['is_ipv4'] = grab('ipv4')

    df_clean['is_http'] = grab('http')
    df_clean['is_https'] = grab('https')
    df_clean['is_dns'] = grab('dns')
    df_clean['is_ssh'] = grab('ssh')
    df_clean['is_telnet'] = grab('telnet')
    df_clean['is_dhcp'] = grab('dhcp')
    df_clean['is_smtp'] = grab('smtp')
    df_clean['is_irc'] = grab('irc')

    if 'protocol' in cols and df_clean['is_tcp'].sum() == 0:
        proto = df[cols['protocol']]
        df_clean['is_tcp'] = (proto == 6).astype(int)
        df_clean['is_udp'] = (proto == 17).astype(int)
        df_clean['is_icmp'] = (proto == 1).astype(int)

    if 'ipv4' not in cols:
        df_clean['is_ipv4'] = 1

    if 'dstport' in cols and df_clean['is_http'].sum() == 0:
        dst_port = df[cols['dstport']]
        df_clean['is_http'] = (dst_port == 80).astype(int)
        df_clean['is_https'] = (dst_port == 443).astype(int)
        df_clean['is_dns'] = (dst_port == 53).astype(int)
        df_clean['is_ssh'] = (dst_port == 22).astype(int)
        df_clean['is_telnet'] = (dst_port == 23).astype(int)
        df_clean['is_dhcp'] = ((dst_port == 67) | (dst_port == 68)).astype(int)
        df_clean['is_smtp'] = (dst_port == 25).astype(int)
        df_clean['is_irc'] = (dst_port == 6667).astype(int)

    df_clean = df_clean.apply(pd.to_numeric, errors='coerce').fillna(0)
    df_clean[df_clean < 0] = 0
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.fillna(0, inplace=True)
    return df_clean


print("Cerberus IDPS -- Phase 3 Validation")
print("Dataset: CIC IoT-DIAD 2024  |  Model: 40-Feature HGBM (Phase 1 & 2 brain)")
print("Zero-shot row uses the loaded model unmodified; later rows are trained on DIAD.")

if not os.path.exists(MODEL_PATH):
    print(f"\nERROR: model file not found at {MODEL_PATH}")
    print("Update MODEL_PATH at the top of this script.")
    exit()

cerberus_model = joblib.load(MODEL_PATH)
expected_features = list(cerberus_model.feature_names_in_)

target_folders = ['Benign', 'DDOS', 'DOS', 'Spoofing']
all_files = []
for folder in target_folders:
    search_path = os.path.join(DIAD_DATA_DIR, folder, "**", "*.csv")
    all_files.extend(glob.glob(search_path, recursive=True))

files_by_class = {0: [], 1: [], 2: []}
for file in all_files:
    file_lower = os.path.basename(file).lower()
    if 'benign' in file_lower: files_by_class[0].append(file)
    elif 'spoof' in file_lower or 'mitm' in file_lower: files_by_class[1].append(file)
    elif 'ddos' in file_lower or 'dos' in file_lower: files_by_class[2].append(file)

final_frames = []
files_failed = []
for target_val, files in files_by_class.items():
    if not files: continue
    budget = CLASS_ROW_BUDGET.get(target_val)
    per_file_limit = max(1, budget // len(files))
    for file in files:
        try:
            df_raw = pd.read_csv(file, nrows=per_file_limit, low_memory=False)
            df_eval = extract_extreme_physics_schema(df_raw)
            df_eval['target_label'] = target_val
            final_frames.append(df_eval)
        except Exception as e:
            files_failed.append((file, str(e)))

class_names = {0: 'Normal', 1: 'MitM', 2: 'DoS'}
file_counts = ", ".join(f"{class_names[c]}: {len(f)}" for c, f in files_by_class.items())

section("Dataset")
print(f"Model expects            {len(expected_features)} features")
print(f"CSV files found          {len(all_files)}  ({file_counts})")
if files_failed:
    print(f"Files skipped (errors)   {len(files_failed)}")

if not final_frames:
    print("\nERROR: no usable files found. Check DIAD_DATA_DIR and folder names.")
    exit()

_diag = pd.concat(final_frames, ignore_index=True)
zero_rates = (_diag.drop(columns=['target_label']) == 0).mean().sort_values(ascending=False)
dead_count = (zero_rates > 0.99).sum()
diag_path = os.path.join(BASE_DIR, 'feature_coverage_diagnostic.csv')
zero_rates.rename('zero_rate').to_csv(diag_path)
print(f"Feature coverage check   {dead_count}/{len(zero_rates)} features >99% sparse in this "
      f"dataset (full breakdown: {diag_path})")

df_master = pd.concat(final_frames, ignore_index=True).drop_duplicates()
min_class_size = df_master['target_label'].value_counts().min()
df_balanced = df_master.groupby('target_label').sample(n=min_class_size, random_state=42)

y_all = df_balanced['target_label']
X_all_raw = df_balanced.drop(columns=['target_label']).astype(np.float32)

X_ft_raw, X_test_raw, y_ft, y_test = train_test_split(
    X_all_raw, y_all, train_size=FINETUNE_FRACTION, stratify=y_all, random_state=42
)

binary_cols = ['is_tcp', 'is_udp', 'is_icmp', 'is_arp', 'is_ipv4',
               'is_http', 'is_https', 'is_dns', 'is_ssh', 'is_telnet',
               'is_dhcp', 'is_smtp', 'is_irc']
cont_cols = [c for c in X_all_raw.columns if c not in binary_cols]

scaler = StandardScaler()
X_ft = X_ft_raw.copy()
X_ft[cont_cols] = scaler.fit_transform(X_ft_raw[cont_cols])
X_test = X_test_raw.copy()
X_test[cont_cols] = scaler.transform(X_test_raw[cont_cols])

missing = set(expected_features) - set(X_ft.columns)
if missing:
    print(f"WARNING: DIAD extraction is missing features the model expects: {missing}")
X_ft = X_ft.reindex(columns=expected_features, fill_value=0)
X_test = X_test.reindex(columns=expected_features, fill_value=0)

print(f"Fine-tune / test split   {len(X_ft_raw):,} / {len(X_test_raw):,} rows")

_t0 = time.perf_counter()
y_pred_before = cerberus_model.predict(X_test)
_inf_time = time.perf_counter() - _t0
latency_us = (_inf_time / len(X_test)) * 1e6
acc_before = accuracy_score(y_test, y_pred_before)
report_before = classification_report(y_test, y_pred_before, target_names=['Normal', 'MitM', 'DoS'], zero_division=0, output_dict=True)

original_max_iter = cerberus_model.max_iter
iters_before_finetune = cerberus_model.n_iter_
sample_weight = np.where(y_ft.isin([0, 1]), 3.0, 1.0)

cerberus_model.set_params(
    warm_start=True,
    max_iter=original_max_iter + FINETUNE_EXTRA_ITERS,
    n_iter_no_change=150,
    validation_fraction=0.15,
)
cerberus_model.fit(X_ft, y_ft, sample_weight=sample_weight)
iters_after_finetune = cerberus_model.n_iter_

section("Fine-Tuning Diagnostic")
print(f"Iterations before        {iters_before_finetune} (of max_iter={original_max_iter})")
print(f"Iterations after         {iters_after_finetune}  "
      f"(+{iters_after_finetune - iters_before_finetune}, budget {FINETUNE_EXTRA_ITERS})")

y_pred_after = cerberus_model.predict(X_test)
acc_after = accuracy_score(y_test, y_pred_after)
report_after = classification_report(y_test, y_pred_after, target_names=['Normal', 'MitM', 'DoS'], zero_division=0, output_dict=True)

fresh_diad_model = HistGradientBoostingClassifier(
    max_iter=1200, learning_rate=0.04, max_leaf_nodes=255, min_samples_leaf=15,
    l2_regularization=0.05, max_bins=255, class_weight='balanced', random_state=42,
    early_stopping=True, validation_fraction=0.125, n_iter_no_change=30
)
fresh_diad_model.fit(X_ft, y_ft)
y_pred_fresh = fresh_diad_model.predict(X_test)
acc_fresh = accuracy_score(y_test, y_pred_fresh)
report_fresh = classification_report(y_test, y_pred_fresh, target_names=['Normal', 'MitM', 'DoS'], zero_division=0, output_dict=True)

section("Results")

results = pd.DataFrame([
    {
        'Variant': 'Zero-Shot (true zero-day)',
        'Accuracy': acc_before,
        'Prec(M)': report_before['macro avg']['precision'],
        'Rec(M)': report_before['macro avg']['recall'],
        'F1(M)': report_before['macro avg']['f1-score'],
        'Prec(W)': report_before['weighted avg']['precision'],
        'Rec(W)': report_before['weighted avg']['recall'],
        'F1(W)': report_before['weighted avg']['f1-score'],
    },
    {
        'Variant': 'Fine-Tuned (adaptation)',
        'Accuracy': acc_after,
        'Prec(M)': report_after['macro avg']['precision'],
        'Rec(M)': report_after['macro avg']['recall'],
        'F1(M)': report_after['macro avg']['f1-score'],
        'Prec(W)': report_after['weighted avg']['precision'],
        'Rec(W)': report_after['weighted avg']['recall'],
        'F1(W)': report_after['weighted avg']['f1-score'],
    },
    {
        'Variant': 'Fresh Retrain (architecture check)',
        'Accuracy': acc_fresh,
        'Prec(M)': report_fresh['macro avg']['precision'],
        'Rec(M)': report_fresh['macro avg']['recall'],
        'F1(M)': report_fresh['macro avg']['f1-score'],
        'Prec(W)': report_fresh['weighted avg']['precision'],
        'Rec(W)': report_fresh['weighted avg']['recall'],
        'F1(W)': report_fresh['weighted avg']['f1-score'],
    },
]).set_index('Variant')

formatted = results.copy()
formatted['Accuracy'] = formatted['Accuracy'].map(lambda v: f"{v*100:.2f}%")
for c in ['Prec(M)', 'Rec(M)', 'F1(M)', 'Prec(W)', 'Rec(W)', 'F1(W)']:
    formatted[c] = formatted[c].map(lambda v: f"{v:.2f}")
print(formatted.to_string())

print(f"\nOnly 'Zero-Shot' is a true zero-day result -- the model was never")
print(f"trained on DIAD for that row. 'Fine-Tuned' and 'Fresh Retrain' are")
print(f"explicitly trained on DIAD data and are adaptation/architecture")
print(f"diagnostics, not validation.")

section("Inference Latency (Zero-Shot, deployed model)")
print(f"Per-packet latency       {latency_us:.2f} microseconds "
      f"({latency_us/1000:.4f} ms)  [target: < 4 ms]")
print(f"Throughput               {len(X_test)/_inf_time:,.0f} packets/second")

for _name, _pred in [("Zero-Shot (true zero-day)", y_pred_before),
                     ("Fine-Tuned (adaptation)", y_pred_after),
                     ("Fresh Retrain (architecture check)", y_pred_fresh)]:
    section(f"Per-Class Detail -- {_name}")
    print(classification_report(y_test, _pred,
          target_names=['Normal', 'MitM', 'DoS'], zero_division=0))

finetuned_path = os.path.join(BASE_DIR, "Cerberus_HGBM_Brain_DIAD_FineTuned.joblib")
joblib.dump(cerberus_model, finetuned_path)
fresh_path = os.path.join(BASE_DIR, "Cerberus_HGBM_Brain_DIAD_Final.joblib")
joblib.dump(fresh_diad_model, fresh_path)

section("Saved")
print(f"Fine-tuned model    {finetuned_path}")
print(f"Fresh model         {fresh_path}")

section("Done")
print("Validation complete. No figures generated by this script.")