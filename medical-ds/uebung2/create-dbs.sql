create or replace table identification_data(
    PID int primary key,
    first_name varchar(256),
    last_name varchar(256),
    birth_year date
);

create or replace table medical_information(
    ID varchar(32),
    Diagnosis varchar(32),
    primary key(ID, Diagnosis)
);

create or replace table relations_table(
    PID int,
    ID varchar(32),
    primary key(PID, ID)
);

create or replace table values_table(
    ID varchar(32),
    MeasurementTime timestamp,
    Measurement varchar(256),
    Value float,
    primary key(ID, MeasurementTime, Measurement)
);
