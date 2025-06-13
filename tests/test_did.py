import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# --- Step 1: Load Data ---
df = pd.read_csv("benchmark/all_data/billboard_impact.csv")

# --- Step 2: Create Interaction Term ---
# poa = 1 for treatment group (Porto Alegre), 0 otherwise
# jul = 1 for post-intervention (July), 0 otherwise
df['did_interaction'] = df['poa'] * df['jul']

# --- Step 3: Specify the DiD Formula ---
# Includes fixed effects for group (poa), time (jul), and their interaction
formula = "deposits ~ did_interaction + C(poa) + C(jul)"

# --- Step 4: Fit the Model ---
model = smf.ols(formula=formula, data=df)
results = model.fit()

# --- Step 5: Extract and Print DiD Estimate ---
coef = results.params['did_interaction']
conf_int = results.conf_int().loc['did_interaction']
stderr = results.bse['did_interaction']
pval = results.pvalues['did_interaction']

print("=== Difference-in-Differences Estimation ===")
print(f"Treatment effect (DiD estimate): {coef:.2f}")
print(f"Standard error: {stderr:.2f}")
print(f"95% CI: ({conf_int[0]:.2f}, {conf_int[1]:.2f})")
print(f"P-value: {pval:.4f}")
print("\nModel Summary:")
print(results.summary())
