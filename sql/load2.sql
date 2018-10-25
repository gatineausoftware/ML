LOAD DATA INFILE '/var/lib/mysql-files/heart.csv'
INTO TABLE health.heart2
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(age,sex,chest_pain_type,resting_blood_pressure,cholesterol,
fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_angina,oldpeak,
slope,colored_vessels,thal,datetime,postalcode,narrowing_diagnosis)