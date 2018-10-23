CREATE DATABASE health

CREATE TABLE health.heart
(
age INT,
sex VARCHAR(10),
chest_pain_type VARCHAR(50),
resting_blood_pressure INT,
cholesterol INT,
fasting_blood_sugar BOOLEAN,
resting_ecg VARCHAR(50),
max_heart_rate INT,
exercise_angina VARCHAR(10),
oldpeak FLOAT,
slope VARCHAR(50),
colored_vessels INT,
thal VARCHAR(50),
datetime DATE,
postalcode VARCHAR(5),
narrowing_diagnosis INT
);

