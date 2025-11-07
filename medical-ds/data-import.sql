TRUNCATE TABLE identification_data; 

LOAD DATA LOCAL INFILE '/Users/luca/Projects/ds/medical-ds/data-for-management/identification_data.csv'
INTO TABLE identification_data
FIELDS TERMINATED BY '","'
LINES STARTING BY '"'
TERMINATED BY '"\n'
IGNORE 1 LINES;


TRUNCATE TABLE medical_information;

LOAD DATA LOCAL INFILE '/Users/luca/Projects/ds/medical-ds/data-for-management/medical_information.csv'
INTO TABLE medical_information
FIELDS TERMINATED BY '","'
LINES STARTING BY '"'
TERMINATED BY '"\n'
IGNORE 1 LINES;

TRUNCATE TABLE relations_table;

LOAD DATA LOCAL INFILE '/Users/luca/Projects/ds/medical-ds/data-for-management/relations_table.csv'
INTO TABLE relations_table
FIELDS TERMINATED BY '","'
LINES STARTING BY '"'
TERMINATED BY '"\n'
IGNORE 1 LINES;

TRUNCATE TABLE values_table;

LOAD DATA LOCAL INFILE '/Users/luca/Projects/ds/medical-ds/data-for-management/values_table.csv'
INTO TABLE values_table
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES STARTING BY '"'
TERMINATED BY '\n'
IGNORE 1 LINES
(@ID, @timestp, Measurement, Value)
SET ID = REPLACE(@ID, '"', ''), MeasurementTime = STR_TO_DATE(@timestp, '%Y-%m-%d %H:%i:%s');
