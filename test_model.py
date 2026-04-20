import joblib
import pandas as pd

# تحميل النموذج
model = joblib.load('model.pkl')

# بيانات الاختبار
test_patients = pd.DataFrame([
    {'age': 25, 'sex': 'female', 'bmi': 22.5, 'children': 0, 'smoker': 'no', 'region': 'southwest'},
    {'age': 45, 'sex': 'male', 'bmi': 32.0, 'children': 2, 'smoker': 'yes', 'region': 'northeast'},
    {'age': 60, 'sex': 'female', 'bmi': 28.5, 'children': 1, 'smoker': 'no', 'region': 'southeast'},
    {'age': 30, 'sex': 'male', 'bmi': 35.0, 'children': 0, 'smoker': 'yes', 'region': 'northwest'},
])

# التوقع
predictions = model.predict(test_patients)

# عرض النتائج
print("\n" + "="*60)
print("📊 PREDICTIONS FOR TEST PATIENTS")
print("="*60)

for i, (_, row) in enumerate(test_patients.iterrows()):
    print(f"\nPatient {i+1}: Age {row['age']}, {row['smoker']}, BMI={row['bmi']}")
    print(f"  → Predicted charge: ${predictions[i]:,.2f}")

print("\n" + "="*60)
print("✅ Test completed!")
