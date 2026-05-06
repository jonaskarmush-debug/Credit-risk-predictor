import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import joblib
import csv
import os
from datetime import datetime

model = joblib.load("models/credit_risk_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

current_theme = "light"

LIGHT_THEME = {
    "bg": "#f5f5f5",
    "fg": "#111111",
    "entry_bg": "white",
    "entry_fg": "#111111",
    "button_bg": "#1f77b4",
    "button_fg": "white"
}

DARK_THEME = {
    "bg": "#1e1e1e",
    "fg": "#f5f5f5",
    "entry_bg": "#2d2d2d",
    "entry_fg": "#f5f5f5",
    "button_bg": "#3a7bd5",
    "button_fg": "white"
}

widgets_to_theme = []
entries_to_theme = []
last_good_probability = 0.5
last_bad_probability = 0.5
last_risk_level = ""
last_decision = ""
last_recommendation = ""


def update_chart(good_probability, bad_probability):
    ax.clear()
    if current_theme == "dark":
        chart_bg = "#1e1e1e"
        text_color = "#f5f5f5"
    else:
        chart_bg = "#f5f5f5"
        text_color = "#111111"
    fig.patch.set_facecolor(chart_bg)
    ax.set_facecolor(chart_bg)
    labels = ["Good Risk", "Bad Risk"]
    values = [good_probability * 100, bad_probability * 100]
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)", fontsize=8, color=text_color, labelpad=6)
    ax.set_title("Credit Risk Probability", color=text_color)
    ax.tick_params(axis="x", colors=text_color, labelsize=8)
    ax.tick_params(axis="y", colors=text_color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(text_color)
    for i, value in enumerate(values):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center", color=text_color)
    fig.tight_layout()
    chart_canvas.draw()


def get_int(entry, field_name):
    value = entry.get().strip().replace(" ", "").replace(",", "")
    if value == "":
        raise ValueError(f"{field_name} is empty.")
    return int(value)


def predict_risk():
    try:
        duration = get_int(duration_entry, "Loan duration")
        if duration < 1 or duration > 120:
            raise ValueError("Loan duration must be between 1 and 120 months")
        credit_amount_sek = get_int(credit_amount_entry, "Loan amount")
        if credit_amount_sek < 1000 or credit_amount_sek > 1000000:
            raise ValueError("Loan amount must be between 1,000 and 1,000,000 SEK")
        current_debt_sek = get_int(current_debt_entry, "Current debt")
        if current_debt_sek < 0 or current_debt_sek > 5000000:
            raise ValueError("Current debt must be between 0 and 5,000,000 SEK")
        credit_amount = credit_amount_sek / 10
        total_debt_after_loan = credit_amount_sek + current_debt_sek
        installment_commitment = get_int(installment_entry, "Repayment burden score")
        if installment_commitment < 1 or installment_commitment > 4:
            raise ValueError("Repayment burden score must be between 1 and 4.")
        residence_since = get_int(residence_entry, "Years at current residence")
        if residence_since < 1 or residence_since > 4:
            raise ValueError("Years at current residence must be between 1 and 4")
        age = get_int(age_entry, "Age")
        if age < 18 or age > 99:
            raise ValueError("Age must be between 18 and 99")
        existing_credits = get_int(existing_credits_entry, "Existing active loans/credits")
        if existing_credits < 0 or existing_credits > 20:
            raise ValueError("Existing active loans/credits must be between 0 and 20")
        num_dependents = get_int(dependents_entry, "Number of dependents")
        if num_dependents < 0 or num_dependents > 10:
            raise ValueError("Number of dependents must be between 0 and 10")

        borrower_encoded = pd.DataFrame([{col: 0 for col in model_columns}])
        borrower_encoded["duration"] = duration
        borrower_encoded["credit_amount"] = credit_amount
        borrower_encoded["installment_commitment"] = installment_commitment
        borrower_encoded["residence_since"] = residence_since
        borrower_encoded["age"] = age
        borrower_encoded["existing_credits"] = existing_credits
        borrower_encoded["num_dependents"] = num_dependents

        categorical_selections = {
            "checking_status": checking_status_var.get(),
            "credit_history": credit_history_var.get(),
            "housing": housing_var.get(),
            "purpose": purpose_var.get(),
            "savings_status": savings_status_var.get(),
            "employment": employment_var.get(),
            "personal_status": "male single",
            "other_parties": "none",
            "property_magnitude": property_var.get(),
            "other_payment_plans": payment_plans_var.get(),
            "job": "skilled",
            "own_telephone": "none",
            "foreign_worker": "yes"
        }

        for col_prefix, value in categorical_selections.items():
            col_name = f"{col_prefix}_{value}"
            match = next((c for c in borrower_encoded.columns if c.strip() == col_name.strip()), None)
            if match:
                borrower_encoded[match] = 1

        bad_probability = model.predict_proba(borrower_encoded)[:, 1][0]
        good_probability = 1 - bad_probability
        prediction = 1 if bad_probability >= 0.50 else 0
        probability_percent = round(bad_probability * 100, 2)

        update_chart(good_probability, bad_probability)

        if bad_probability < 0.25:
            risk_level = "LOW RISK"
            risk_color = "green"
        elif bad_probability < 0.50:
            risk_level = "MEDIUM RISK"
            risk_color = "orange"
        elif bad_probability < 0.75:
            risk_level = "HIGH RISK"
            risk_color = "red"
        else:
            risk_level = "VERY HIGH RISK"
            risk_color = "darkred"

        model_decision = "BAD CREDIT RISK" if prediction == 1 else "GOOD CREDIT RISK"

        if bad_probability < 0.25:
            recommendation = "Recommendation: APPROVE\nLow risk borrower."
        elif bad_probability < 0.50:
            recommendation = "Recommendation: APPROVE WITH CAUTION\nMonitor repayments closely."
        elif bad_probability < 0.75:
            recommendation = "Recommendation: REVIEW CAREFULLY\nConsider collateral or co-signer."
        else:
            recommendation = "Recommendation: DECLINE\nVery high probability of default."

        result_text = (
            f"{model_decision}\n"
            f"Risk level: {risk_level}\n"
            f"Probability of bad risk: {probability_percent}%\n"
            f"Total debt after new loan: {total_debt_after_loan:,} SEK\n\n"
            f"{recommendation}"
        )

        global last_good_probability, last_bad_probability, last_risk_level, last_decision, last_recommendation
        last_good_probability = good_probability
        last_bad_probability = bad_probability
        last_risk_level = risk_level
        last_decision = model_decision
        last_recommendation = recommendation

        result_label.config(text=result_text, fg=risk_color)
        show_tab("summary")

        drivers = []
        if duration > 36:
            drivers.append("⚠ Long loan duration")
        if installment_commitment >= 3:
            drivers.append("⚠ High repayment burden")
        if savings_status_var.get() in ["<100", "no known savings"]:
            drivers.append("⚠ Low savings status")
        if checking_status_var.get() in ["<0", "no checking"]:
            drivers.append("⚠ Weak checking account")
        if credit_history_var.get() in ["delayed previously", "all paid", "no credits/all paid"]:
            drivers.append("⚠ Problematic credit history")
        if existing_credits >= 2:
            drivers.append("⚠ Multiple existing credits")
        if num_dependents >= 2:
            drivers.append("⚠ High number of dependents")

        drivers_text = "\n".join(drivers) if drivers else "✓ No major risk drivers identified"
        summary_text.config(text=f"Top Risk Drivers:\n\n{drivers_text}", fg=risk_color)

        borrower_count[0] += 1
        history_listbox.insert(
            tk.END,
            f"#{borrower_count[0]}  {model_decision}  {probability_percent}%  Age:{age}  Loan:{credit_amount_sek:,} SEK"
        )

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))


def save_prediction():
    if last_decision == "":
        messagebox.showwarning("No Prediction", "Please make a prediction first!")
        return
    filepath = "predictions_log.csv"
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Date/Time", "Loan Amount (SEK)", "Current Debt (SEK)", "Age",
                "Credit History", "Checking Status", "Housing",
                "Bad Risk %", "Good Risk %", "Risk Level", "Decision", "Recommendation"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            credit_amount_entry.get(),
            current_debt_entry.get(),
            age_entry.get(),
            credit_history_var.get(),
            checking_status_var.get(),
            housing_var.get(),
            f"{last_bad_probability * 100:.2f}%",
            f"{last_good_probability * 100:.2f}%",
            last_risk_level,
            last_decision,
            last_recommendation
        ])
    messagebox.showinfo("Saved", "Prediction saved to predictions_log.csv!")


def show_tab(tab_name):
    if tab_name == "summary":
        history_frame.pack_forget()
        summary_frame.pack(pady=10, fill="both", expand=True)
    else:
        summary_frame.pack_forget()
        history_frame.pack(pady=10, fill="both", expand=True)


def apply_theme(theme):
    root.config(bg=theme["bg"])
    top_frame.config(bg=theme["bg"])
    columns_frame.config(bg=theme["bg"])
    left_frame.config(bg=theme["bg"])
    right_frame.config(bg=theme["bg"])
    tab_frame.config(bg=theme["bg"])
    tab_button_frame.config(bg=theme["bg"])
    summary_frame.config(bg=theme["bg"])
    history_frame.config(bg=theme["bg"])
    for widget in widgets_to_theme:
        try:
            widget.config(bg=theme["bg"], fg=theme["fg"])
        except Exception:
            pass
    for entry in entries_to_theme:
        entry.config(bg=theme["entry_bg"], fg=theme["entry_fg"], insertbackground=theme["fg"])
    predict_button.config(bg=theme["button_bg"], fg=theme["button_fg"])
    theme_button.config(bg=theme["button_bg"], fg=theme["button_fg"])
    try:
        history_listbox.config(bg=theme["entry_bg"], fg=theme["fg"])
    except:
        pass
    update_chart(last_good_probability, last_bad_probability)


def toggle_theme():
    global current_theme
    if current_theme == "light":
        current_theme = "dark"
        apply_theme(DARK_THEME)
        theme_button.config(text="Switch to Light Mode")
    else:
        current_theme = "light"
        apply_theme(LIGHT_THEME)
        theme_button.config(text="Switch to Dark Mode")


def add_input(label_text, default_value):
    label = tk.Label(left_frame, text=label_text, anchor="w", width=28)
    label.pack()
    widgets_to_theme.append(label)
    entry = tk.Entry(left_frame, width=30)
    entry.insert(0, default_value)
    entry.pack(pady=4)
    entries_to_theme.append(entry)
    return entry


def add_dropdown(label_text, variable, values):
    label = tk.Label(right_frame, text=label_text, anchor="w", width=28)
    label.pack()
    widgets_to_theme.append(label)
    dropdown = ttk.Combobox(right_frame, textvariable=variable, values=values, state="readonly", width=28)
    dropdown.pack(pady=4)
    return dropdown


# ── Main window ───────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Credit Risk Predictor")
root.geometry("1400x900")
root.state("zoomed")
root.resizable(True, True)

# Top section
top_frame = tk.Frame(root)
top_frame.pack(side="top", fill="x", pady=10)

title_label = tk.Label(top_frame, text="Credit Risk Predictor", font=("Arial", 18, "bold"))
title_label.pack()
widgets_to_theme.append(title_label)

subtitle_label = tk.Label(top_frame, text="Enter borrower information and predict credit risk.", font=("Arial", 10))
subtitle_label.pack(pady=4)
widgets_to_theme.append(subtitle_label)

chart_frame = tk.Frame(top_frame)
chart_frame.pack()

fig, ax = plt.subplots(figsize=(5, 2.8))
chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
chart_canvas.get_tk_widget().pack()
update_chart(0.5, 0.5)

# Middle columns
columns_frame = tk.Frame(root)
columns_frame.pack(fill="both", expand=True, pady=5)

left_frame = tk.Frame(columns_frame)
left_frame.pack(side="left", padx=20, anchor="n")

right_frame = tk.Frame(columns_frame)
right_frame.pack(side="left", padx=20, anchor="n")

tab_frame = tk.Frame(columns_frame)
tab_frame.pack(side="left", padx=20, anchor="n", fill="both", expand=True)

# Left column inputs
left_title = tk.Label(left_frame, text="Borrower Details", font=("Arial", 11, "bold"))
left_title.pack(pady=(0, 8))
widgets_to_theme.append(left_title)

duration_entry         = add_input("Loan duration (months)", "24")
credit_amount_entry    = add_input("Loan amount (SEK)", "30000")
current_debt_entry     = add_input("Current debt (SEK)", "0")
installment_entry      = add_input("Repayment burden score (1-4)", "2")
residence_entry        = add_input("Years at current residence (1-4)", "2")
age_entry              = add_input("Age", "35")
existing_credits_entry = add_input("Existing active loans/credits", "1")
dependents_entry       = add_input("Number of dependents", "1")

result_label = tk.Label(left_frame, text="", font=("Arial", 11, "bold"), wraplength=250, justify="left")
result_label.pack(pady=10)
widgets_to_theme.append(result_label)

# Right column dropdowns
right_title = tk.Label(right_frame, text="Borrower Profile", font=("Arial", 11, "bold"))
right_title.pack(pady=(0, 8))
widgets_to_theme.append(right_title)

housing_var         = tk.StringVar(value="own")
checking_status_var = tk.StringVar(value="no checking")
credit_history_var  = tk.StringVar(value="existing paid")
savings_status_var  = tk.StringVar(value="<100")
employment_var      = tk.StringVar(value="1<=X<4")
purpose_var         = tk.StringVar(value="radio/tv")
property_var        = tk.StringVar(value="real estate")
payment_plans_var   = tk.StringVar(value="none")

add_dropdown("Housing", housing_var, ["own", "rent", "for free"])
add_dropdown("Checking account status", checking_status_var, ["<0", "0<=X<200", ">=200", "no checking"])
add_dropdown("Credit history", credit_history_var, [
    "existing paid", "critical/other existing credit",
    "delayed previously", "all paid", "no credits/all paid"
])
add_dropdown("Savings status", savings_status_var, ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"])
add_dropdown("Employment length", employment_var, ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"])
add_dropdown("Loan purpose", purpose_var, [
    "new car", "used car", "furniture/equipment", "radio/tv",
    "domestic appliance", "repairs", "education", "vacation", "retraining", "business", "other"
])
add_dropdown("Property / collateral", property_var, ["real estate", "life insurance", "car", "no known property"])
add_dropdown("Other payment plans", payment_plans_var, ["bank", "stores", "none"])

# Tab area
tab_button_frame = tk.Frame(tab_frame)
tab_button_frame.pack(pady=(0, 5))

risk_summary_btn = tk.Button(
    tab_button_frame, text="Risk Summary",
    font=("Arial", 10, "bold"), width=14,
    command=lambda: show_tab("summary")
)
risk_summary_btn.pack(side="left", padx=4)

history_btn = tk.Button(
    tab_button_frame, text="Borrower History",
    font=("Arial", 10, "bold"), width=14,
    command=lambda: show_tab("history")
)
history_btn.pack(side="left", padx=4)

predict_button = tk.Button(
    tab_button_frame, text="Predict Credit Risk", command=predict_risk,
    font=("Arial", 9, "bold"), bg="#1f77b4", fg="white", width=16
)
predict_button.pack(side="left", padx=4)

theme_button = tk.Button(
    tab_button_frame, text="Switch to Dark Mode", command=toggle_theme,
    font=("Arial", 9, "bold"), bg="#444444", fg="white", width=16
)
theme_button.pack(side="left", padx=4)

save_button = tk.Button(
    tab_button_frame, text="Save Prediction", command=save_prediction,
    font=("Arial", 9, "bold"), bg="#2ecc71", fg="white", width=14
)
save_button.pack(side="left", padx=4)

summary_frame = tk.Frame(tab_frame)
history_frame = tk.Frame(tab_frame)

summary_title = tk.Label(summary_frame, text="Risk Summary", font=("Arial", 12, "bold"))
summary_title.pack(pady=(10, 5))
widgets_to_theme.append(summary_title)

summary_text = tk.Label(
    summary_frame, text="Press 'Predict Credit Risk' to see summary.",
    font=("Arial", 10), wraplength=280, justify="left"
)
summary_text.pack(pady=5)
widgets_to_theme.append(summary_text)

history_title = tk.Label(history_frame, text="Borrower History", font=("Arial", 12, "bold"))
history_title.pack(pady=(10, 5))
widgets_to_theme.append(history_title)

history_listbox = tk.Listbox(history_frame, width=45, height=15, font=("Courier", 9))
history_listbox.pack(pady=5)

clear_history_btn = tk.Button(
    history_frame, text="Clear History",
    font=("Arial", 9), width=16,
    command=lambda: history_listbox.delete(0, tk.END)
)
clear_history_btn.pack(pady=4)

borrower_count = [0]

apply_theme(LIGHT_THEME)
root.mainloop()
