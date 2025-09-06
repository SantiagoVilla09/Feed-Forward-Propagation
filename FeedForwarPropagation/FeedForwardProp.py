import os
import sys
import numpy as np
import pandas as pd

# ----------------------------
# CONFIGURACIÓN (ajustable)
# ----------------------------
CSV_NAME = "mushrooms.csv"   # archivo original
TARGET_COLUMN = "class"      # en mushrooms la salida es 'class'
POSITIVE_LABEL = "p"         # 1 si 'p' (poisonous), 0 en otro caso ('e' edible)

TEST_SIZE = 0.2              # 20% para prueba, 80% para entrenamiento
RANDOM_SEED = 42             # reproducibilidad

# Hiperparámetros del árbol base
MAX_DEPTH = 8
MIN_SAMPLES_SPLIT = 15
MIN_SAMPLES_LEAF = 10
MIN_GAIN = 1e-4

# Ensamble (bagging + random subspace)
N_TREES = 35
FEATURE_FRACTION = 0.6
BOOTSTRAP_FRACTION = 1.0

# Umbral fijo
PRED_THRESHOLD = 0.5

# ============================================================
# UTILIDADES DE LECTURA Y SPLIT
# ============================================================
def _clean_header(name: str) -> str:
    """Limpia caracteres raros en encabezados (BOM, NBSP) y espacios."""
    return str(name).replace("\ufeff", "").replace("\xa0", " ").strip()

def _find_target_column(cols):
    """Encuentra TARGET_COLUMN ignorando espacios y guiones bajos."""
    norm = {_clean_header(c).lower().replace(" ", "").replace("_",""): c for c in cols}
    key = TARGET_COLUMN.lower().replace(" ", "").replace("_","")
    return norm.get(key, None)

def read_csv_autosep(path: str) -> pd.DataFrame:
    """Intenta leer el CSV probando separadores comunes."""
    for sep in [";", ",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    # Último intento: dejar que pandas infiera
    return pd.read_csv(path, encoding="utf-8-sig")

def coerce_categorical_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fuerza todas las columnas a tipo string (categóricas) y limpia nulos."""
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(str).fillna("Unknown").str.strip()
    return out

def stratified_train_test_split_df(df: pd.DataFrame, label_col: str, test_size: float, seed: int):
    """
    Split estratificado manual:
    - Mantiene las proporciones de clases entre train y test.
    - Baraja por clase para no sesgar.
    """
    rng = np.random.default_rng(seed)
    parts_tr, parts_te = [], []
    for _, grp in df.groupby(label_col):
        idx = np.arange(len(grp))
        rng.shuffle(idx)
        n_test = max(1, min(int(round(len(grp)*test_size)), len(grp)-1))
        parts_te.append(grp.iloc[idx[:n_test]])
        parts_tr.append(grp.iloc[idx[n_test:]])
    tr = pd.concat(parts_tr).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    te = pd.concat(parts_te).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return tr, te

# ============================================================
# ÁRBOL (criterio Gini) – categórico y numérico (por si acaso)
# ============================================================
def _is_numeric(s): 
    return pd.api.types.is_numeric_dtype(s)

def _class_counts(y): 
    return np.bincount(y, minlength=int(np.max(y))+1 if y.size else 1)

def _majority(y): 
    return int(np.argmax(_class_counts(y)))

def _class_dist(y):
    cnt = _class_counts(y); tot = int(cnt.sum())
    return {int(i): float(c)/tot for i, c in enumerate(cnt)} if tot else {0:0.0,1:0.0}

def gini_from_counts(counts): 
    tot = counts.sum()
    if tot == 0: 
        return 0.0
    p = counts / tot
    return 1.0 - float(np.sum(p*p))

def gini(y): 
    return gini_from_counts(_class_counts(y))

def gini_gain_numeric(y, x: pd.Series):
    """
    Split numérico mediante un conjunto pequeño de umbrales (cuantiles).
    Aunque mushrooms es categórico, dejamos soporte numérico por robustez.
    """
    xv = x.values.astype(float)
    if len(xv) <= 1: 
        return -1.0, None
    qs = np.linspace(0.1, 0.9, 9)
    cand = np.unique(np.quantile(xv, qs, method="linear"))
    if cand.size == 0: 
        return -1.0, None
    base, n = gini(y), len(y)
    best_gain, best_t = -1.0, None
    for t in cand:
        left_idx = xv <= t
        nl, nr = int(np.sum(left_idx)), n - int(np.sum(left_idx))
        if nl < MIN_SAMPLES_LEAF or nr < MIN_SAMPLES_LEAF: 
            continue
        yl, yr = y[left_idx], y[~left_idx]
        gain = base - (nl/n)*gini(yl) - (nr/n)*gini(yr)
        if gain > best_gain:
            best_gain, best_t = float(gain), float(t)
    return best_gain, best_t

def gini_gain_categorical(y, x: pd.Series):
    """
    Split categórico multi-rama por valor.
    - Penaliza valores con muy pocas filas para evitar hojas minúsculas.
    """
    xv = x.values
    vals = pd.unique(xv)
    for v in vals:
        if np.sum(xv == v) < MIN_SAMPLES_LEAF:
            return -1.0, None
    base, n = gini(y), len(y)
    weighted = 0.0
    split_map = {}
    for v in vals:
        mask = (xv == v)
        weighted += (np.sum(mask)/n) * gini(y[mask])
        split_map[v] = mask
    gain = base - weighted
    return float(gain), split_map

def build_tree_gini(X_df: pd.DataFrame, y: np.ndarray, depth: int = 0):
    """Construye recursivamente un árbol de decisión con criterio Gini."""
    node = {"n_samples": int(len(y)), "depth": int(depth)}
    uniq = np.unique(y)
    if len(uniq) == 1:
        node.update({"leaf": True, "prediction": int(uniq[0]), "probs": _class_dist(y)})
        return node
    if depth >= MAX_DEPTH or len(y) < MIN_SAMPLES_SPLIT or X_df.shape[1] == 0:
        node.update({"leaf": True, "prediction": _majority(y), "probs": _class_dist(y)})
        return node

    best_gain, best_feat, best_info, best_is_num, best_thr = -1.0, None, None, None, None

    # Buscar el mejor split entre las columnas disponibles
    for feat in X_df.columns:
        s = X_df[feat]
        if _is_numeric(s):
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().sum() >= (len(s) * 0.7):
                gain, thr = gini_gain_numeric(y, s_num)
                info = {"threshold": thr}
                isnum = True
            else:
                gain, split_map = gini_gain_categorical(y, s.astype(str))
                info = {"children": split_map}
                isnum = False
        else:
            gain, split_map = gini_gain_categorical(y, s)
            info = {"children": split_map}
            isnum = False

        if gain > best_gain:
            best_gain, best_feat, best_info, best_is_num = gain, feat, info, isnum
            best_thr = info.get("threshold", None) if isnum else None

    if best_feat is None or best_gain <= MIN_GAIN:
        node.update({"leaf": True, "prediction": _majority(y), "probs": _class_dist(y)})
        return node

    node.update({
        "leaf": False,
        "feature": best_feat,
        "is_numeric": bool(best_is_num),
        "default": _majority(y)
    })

    if best_is_num:
        node["threshold"] = float(best_thr)
        left_mask = pd.to_numeric(X_df[best_feat], errors="coerce").fillna(np.nan).values <= best_thr
        Xl, Xr = X_df.loc[left_mask], X_df.loc[~left_mask]
        yl, yr = y[left_mask], y[~left_mask]
        if len(yl) < MIN_SAMPLES_LEAF or len(yr) < MIN_SAMPLES_LEAF:
            node.update({"leaf": True, "prediction": _majority(y), "probs": _class_dist(y)})
            return node
        node["left"]  = build_tree_gini(Xl, yl, depth+1)
        node["right"] = build_tree_gini(Xr, yr, depth+1)
    else:
        node["children"] = {}
        for val, mask in best_info["children"].items():
            X_child = X_df.loc[mask].drop(columns=[best_feat])
            y_child = y[mask]
            if len(y_child) < MIN_SAMPLES_LEAF:
                node["children"][str(val)] = {
                    "leaf": True,
                    "prediction": _majority(y_child),
                    "probs": _class_dist(y_child)
                }
            else:
                node["children"][str(val)] = build_tree_gini(X_child, y_child, depth+1)

    return node

def predict_proba_row(row: pd.Series, node: dict) -> float:
    """Recorre el árbol para una fila y devuelve P(y=1)."""
    while not node.get("leaf", False):
        feat = node["feature"]
        default = node["default"]
        if node["is_numeric"]:
            val = row.get(feat, np.nan)
            try:
                val = float(val)
            except Exception:
                return 1.0 if default == 1 else 0.0
            node = node["left"] if val <= node["threshold"] else node["right"]
        else:
            key = row.get(feat, None)
            child = node["children"].get(str(key))
            if child is None:
                return 1.0 if default == 1 else 0.0
            node = child
    probs = node.get("probs", {})
    return float(probs.get(1, probs.get("1", 0.0)))

def predict_proba_df(X_df: pd.DataFrame, tree: dict) -> np.ndarray:
    """Predicción de probabilidades para todas las filas de X."""
    return np.array([predict_proba_row(row, tree) for _, row in X_df.iterrows()], dtype=float)

# ============================================================
# ENSAMBLE (bagging + random subspace)
# ============================================================
def bootstrap_sample(X: pd.DataFrame, y: np.ndarray, frac: float, rng):
    """Bootstrap con reemplazo de tamaño 'frac' del dataset."""
    n = len(X); m = max(1, int(round(frac * n)))
    idx = rng.integers(0, n, size=m)
    return X.iloc[idx], y[idx]

def choose_feature_subset(cols, frac, rng):
    """Selecciona aleatoriamente un subconjunto de columnas para un árbol."""
    k = max(1, int(round(len(cols) * frac)))
    idx = rng.choice(len(cols), size=k, replace=False)
    return [cols[i] for i in idx]

def train_one_tree(X: pd.DataFrame, y: np.ndarray, rng):
    """Entrena un árbol sobre un bootstrap y un subespacio de features."""
    cols = list(X.columns)
    sub_cols = choose_feature_subset(cols, FEATURE_FRACTION, rng)
    X_sub = X[sub_cols]
    X_boot, y_boot = bootstrap_sample(X_sub, y, BOOTSTRAP_FRACTION, rng)
    tree = build_tree_gini(X_boot, y_boot, depth=0)
    return {"tree": tree, "cols": sub_cols}

def ensemble_predict_proba(X: pd.DataFrame, forest: list[dict]) -> np.ndarray:
    """Promedia las probabilidades reportadas por cada árbol del ensamble."""
    acc = np.zeros(len(X), dtype=float)
    for t in forest:
        acc += predict_proba_df(X[t["cols"]], t["tree"])
    return acc / len(forest)

# ============================================================
# MÉTRICAS
# ============================================================
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """Devuelve TP, TN, FP, FN como diccionario."""
    tp = int(np.sum((y_true==1)&(y_pred==1)))
    tn = int(np.sum((y_true==0)&(y_pred==0)))
    fp = int(np.sum((y_true==0)&(y_pred==1)))
    fn = int(np.sum((y_true==1)&(y_pred==0)))
    return {"TP":tp,"TN":tn,"FP":fp,"FN":fn}

def metrics_acc_prec(y_true: np.ndarray, y_pred: np.ndarray):
    """Calcula accuracy y precision a partir de la matriz de confusión."""
    cm = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    tot = tp + tn + fp + fn
    acc = (tp + tn) / max(1, tot)
    prec = tp / max(1, tp + fp)
    return {"accuracy":acc, "precision":prec}, cm

# ============================================================
# MAIN: split -> guarda CSVs -> entrena -> evalúa
# ============================================================
if __name__=="__main__":
    # Permite pasar la ruta por argumento, o usa CSV_NAME por defecto
    csv_path = CSV_NAME if len(sys.argv) < 2 else sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"No se encontró el CSV: {csv_path}")
        sys.exit(1)

    # 1) Cargar dataset original y limpiar encabezados
    raw_df = read_csv_autosep(csv_path)
    raw_df.columns = [_clean_header(c) for c in raw_df.columns]

    # 2) Confirmar y normalizar target
    tgt = _find_target_column(raw_df.columns)
    if tgt is None:
        raise ValueError(f"No se encontró '{TARGET_COLUMN}'. Columnas: {list(raw_df.columns)}")

    # 3) Forzar todo a categórico (string)
    df = coerce_categorical_df(raw_df)

    # 4) Guardar split 80/20 en CSVs (estratificado por el target original)
    #    Se guardan ANTES de entrenar, como pediste.
    base = os.path.splitext(os.path.basename(csv_path))[0]
    train_csv = f"{base}_train.csv"
    test_csv  = f"{base}_test.csv"

    train_df, test_df = stratified_train_test_split_df(df, tgt, TEST_SIZE, RANDOM_SEED)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("=== Splits guardados ===")
    print(f"  Train -> {train_csv}  (filas: {len(train_df)})")
    print(f"  Test  -> {test_csv}   (filas: {len(test_df)})")

    # 5) Preparar X, y EN FUNCIÓN DEL SPLIT GUARDADO
    #    (entrenamos y evaluamos usando exclusivamente los CSVs recién escritos)
    train_df = pd.read_csv(train_csv, dtype=str).applymap(lambda x: x.strip() if isinstance(x,str) else x)
    test_df  = pd.read_csv(test_csv,  dtype=str).applymap(lambda x: x.strip() if isinstance(x,str) else x)

    # 6) Binarizar y (1 si POSITIVE_LABEL, 0 si no)
    y_tr = (train_df[tgt].astype(str).str.strip() == str(POSITIVE_LABEL)).astype(int).values
    y_te = (test_df[tgt].astype(str).str.strip()  == str(POSITIVE_LABEL)).astype(int).values

    # 7) Separar features
    X_tr = train_df.drop(columns=[tgt])
    X_te = test_df.drop(columns=[tgt])

    # 8) Entrenar ensamble con el split de TRAIN
    rng = np.random.default_rng(RANDOM_SEED)
    forest = [train_one_tree(X_tr, y_tr, rng) for _ in range(N_TREES)]

    # 9) Evaluar en TEST
    proba_te = ensemble_predict_proba(X_te, forest)
    y_hat = (proba_te >= PRED_THRESHOLD).astype(int)
    met, cm = metrics_acc_prec(y_te, y_hat)

    # 10) Guardar artefactos de evaluación para tu reporte
    pd.DataFrame([cm]).to_csv("confusion_matrix.csv", index=False)
    pd.DataFrame([{
        "accuracy": met["accuracy"],
        "precision": met["precision"],
        "pred_threshold": PRED_THRESHOLD
    }]).to_csv("metrics.csv", index=False)

    # 11) Mostrar resumen
    print("\n=== Entrenamiento y evaluación ===")
    print(f"threshold usado: {PRED_THRESHOLD}")
    print("Confusion Matrix (test):", cm)
    print("Metrics (test): acc={:.4f} prec={:.4f}".format(met["accuracy"], met["precision"]))
    print("\nGuardados:\n  -", train_csv, "\n  -", test_csv, "\n  - confusion_matrix.csv\n  - metrics.csv")
