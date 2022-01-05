-- This SQL file sets up databases and users across the different components.

-- Pilot.
DROP DATABASE IF EXISTS np_pilot;
DROP USER IF EXISTS np_pilot_daemon;
DROP USER IF EXISTS np_pilot_client;

CREATE DATABASE np_pilot;
CREATE USER np_pilot_daemon WITH ENCRYPTED PASSWORD 'np_pilot_daemon_pass';
GRANT ALL PRIVILEGES ON DATABASE np_pilot TO np_pilot_daemon;
CREATE USER np_pilot_client WITH ENCRYPTED PASSWORD 'np_pilot_client_pass';
GRANT ALL PRIVILEGES ON DATABASE np_pilot TO np_pilot_client;

-- Action selection: OpenSpiel.
DROP DATABASE IF EXISTS np_as_spiel;
DROP USER IF EXISTS np_as_spiel_user;
CREATE DATABASE np_as_spiel;
CREATE USER np_as_spiel_user WITH ENCRYPTED PASSWORD 'np_as_spiel_user_pass';
GRANT ALL PRIVILEGES ON DATABASE np_as_spiel to np_as_spiel_user;
