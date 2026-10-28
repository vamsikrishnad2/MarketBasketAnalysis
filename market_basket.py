
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Sample shopping transactions
transactions = [
    ["Bread", "Milk", "Butter"],
    ["Bread", "Milk"],
    ["Bread", "Butter"],
    ["Milk", "Butter"],
    ["Bread", "Milk", "Butter"],
    ["Bread", "Eggs"],
    ["Milk", "Eggs"],
    ["Bread", "Milk", "Eggs"],
    ["Bread", "Milk", "Butter"],
    ["Bread", "Butter", "Eggs"]
]

# Step 2: Convert transactions into a table
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

print("TRANSACTION DATA")
print(df)

# Step 3: Find frequent product combinations
frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\nFREQUENT ITEMSETS")
print(frequent_itemsets)

# Step 4: Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print("\nASSOCIATION RULES")
print(rules[["antecedents", "consequents",
             "support", "confidence", "lift"]])

# Step 5: Plot frequent product combinations
if not frequent_itemsets.empty:
    plot_data = frequent_itemsets.copy()
    plot_data["items"] = plot_data["itemsets"].apply(
        lambda items: ", ".join(sorted(items))
    )

    plot_data = plot_data.sort_values(
        "support", ascending=False
    )

    plt.figure(figsize=(10, 5))
    plt.bar(plot_data["items"], plot_data["support"])
    plt.xlabel("Product combinations")
    plt.ylabel("Support")
    plt.title("Frequent Product Combinations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("output/frequent_itemsets.png")
    plt.show()
else:
    print("No frequent itemsets found.")
