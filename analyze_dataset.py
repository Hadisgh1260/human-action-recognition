import os
from collections import Counter
import matplotlib.pyplot as plt

# مسیر پوشه‌ی دیتاست
base_path = r'Action Recognition.yolo26'
class_names = ['Run', 'Sit', 'Stand', 'Walk']
splits = ['train', 'valid', 'test']

# شمارش تعداد هر کلاس در هر split
counts = {split: Counter() for split in splits}

for split in splits:
    labels_path = os.path.join(base_path, split, 'labels')
    if not os.path.exists(labels_path):
        print(f"Warning: {labels_path} not found")
        continue
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_path, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        counts[split][class_names[class_id]] += 1

# چاپ خلاصه توی ترمینال
print("\n--- Class Distribution ---")
for split in splits:
    print(f"\n{split.upper()}:")
    for cls in class_names:
        print(f"  {cls}: {counts[split][cls]}")

# رسم نمودار
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(class_names))
width = 0.25

for i, split in enumerate(splits):
    values = [counts[split][cls] for cls in class_names]
    ax.bar([p + i * width for p in x], values, width, label=split)

ax.set_xlabel('Class')
ax.set_ylabel('Number of instances')
ax.set_title('Class Distribution Across Train/Valid/Test')
ax.set_xticks([p + width for p in x])
ax.set_xticklabels(class_names)
ax.legend()

plt.tight_layout()
plt.savefig('class_distribution.png')
print("\nChart saved as class_distribution.png")
plt.show()