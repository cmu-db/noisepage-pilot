-- This sets up the tables for the Pilot component.
DROP TABLE IF EXISTS pilot_results;
DROP TABLE IF EXISTS pilot_commands;
DROP TYPE IF EXISTS pilot_cmd_type;
DROP TYPE IF EXISTS pilot_cmd_status;

-- INPUT TABLE: pilot_commands.
-- The Pilot client inserts rows into this table to send commands.
CREATE TYPE pilot_cmd_type AS ENUM ('cmd_run', 'cmd_abort');
CREATE TABLE IF NOT EXISTS pilot_commands
(
    id    SERIAL PRIMARY KEY,
    ts    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ctype pilot_cmd_type,
    args  TEXT
);

-- OUTPUT TABLE: pilot_results.
-- The Pilot daemon writes results of completed commands into this table.
CREATE TYPE pilot_cmd_status AS ENUM ('running', 'success', 'aborted');
CREATE TABLE IF NOT EXISTS pilot_results
(
    id     SERIAL PRIMARY KEY,
    ts     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status pilot_cmd_status,
    cid    INT REFERENCES pilot_commands (id),
    output TEXT
);

-- notify_pilot() is a trigger that should be invoked
-- to notify the pilot daemon (Python) of new commands
-- that should be performed.
CREATE OR REPLACE FUNCTION pilot_notify() RETURNS TRIGGER AS
$$
BEGIN
    -- NEW = new database row for INSERT in row trigger.
    -- For more info, see: https://www.postgresql.org/docs/14/plpgsql-trigger.html
    PERFORM pg_notify(
            'pilot',
            NEW.id::text || ',' || NEW.ctype::text || ',' || NEW.args::text
        );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Register the pilot_notify() trigger to be run every time a row is
-- inserted into pilot_commands.
CREATE
OR
REPLACE
TRIGGER cmd_notify AFTER
INSERT
ON pilot_commands FOR EACH ROW
EXECUTE PROCEDURE pilot_notify();

-- TODO(WAN): hardcoded np_pilot_client user.
-- Grant permissions.
GRANT USAGE, SELECT ON SEQUENCE pilot_commands_id_seq TO np_pilot_client;
GRANT INSERT ON pilot_commands TO np_pilot_client;
GRANT SELECT ON pilot_results TO np_pilot_client;
