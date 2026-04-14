import pandas as pd
import numpy as np

np.random.seed(42)

# ======================
# PARAMETER
# ======================
n = 1500  # jumlah nasabah

# ======================
# GENERATE DATA
# ======================
data = pd.DataFrame({
    "customer_id": range(1, n+1),
    
    "job_type": np.random.choice(
        ["Karyawan", "Wiraswasta", "Freelance", "UMKM"],
        n
    ),
    
    "monthly_income": np.random.randint(2000000, 15000000, n),
    
    "loan_amount": np.random.randint(500000, 10000000, n),
    
    "collateral_value": np.random.randint(1000000, 15000000, n),
    
    "loan_count": np.random.randint(1, 10, n),
    
    "days_late": np.random.randint(0, 30, n),
    
    "branch": np.random.choice(
        ["Jakarta", "Bandung", "Surabaya", "Medan"],
        n
    )
})

# ======================
# LOGIC REDEEMED (REALISTIS)
# ======================
def generate_redeemed(row):
    risk = 0
    
    # LTV tinggi → risk naik
    if row["loan_amount"] / row["collateral_value"] > 0.8:
        risk += 1
        
    # telat → risk naik
    if row["days_late"] > 7:
        risk += 1
        
    # income rendah → risk naik
    if row["monthly_income"] < 4000000:
        risk += 1

    # probabilitas gagal bayar
    return 0 if risk >= 2 else 1

data["redeemed"] = data.apply(generate_redeemed, axis=1)

# ======================
# SAVE
# ======================
data.to_csv("data/pawn_data.csv", index=False)

print("✅ Data berhasil dibuat:", len(data), "nasabah")
