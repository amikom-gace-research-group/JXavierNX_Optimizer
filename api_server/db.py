import sqlite3

def get_db(port):
    conn = sqlite3.connect(f"database-{port}.db")
    return conn

def create_table_readkey(port):
    tables = [
            """CREATE TABLE IF NOT EXISTS
                readkey_table(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL)"""
        ]
    
    db = get_db(port)
    cursor = db.cursor()
    
    for table in tables:
        cursor.execute(table)

def create_table_account(port):
    tables = [
            """CREATE TABLE IF NOT EXISTS
                account(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    password TEXT NOT NULL)"""
        ]

    db = get_db(port)
    cursor = db.cursor()

    for table in tables:
        cursor.execute(table)

def create_table_writekey(port):
    tables = [
           """CREATE TABLE IF NOT EXISTS
                writekey_table(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL)"""
        ]
    
    db = get_db(port)
    cursor = db.cursor()
    
    for table in tables:
        cursor.execute(table)
    
def create_table_cfg(port):
    tables = [
           """CREATE TABLE IF NOT EXISTS
                cfg_table(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cpu_cores INTEGER NOT NULL,
                    cpu_freq INTEGER NOT NULL,
                    gpu_freq INTEGER NOT NULL,
                    mem_freq INTEGER NOT NULL,
                    cl INTEGER NOT NULL
                    )"""
        ]
    
    db = get_db(port)
    cursor = db.cursor()
    
    for table in tables:
        cursor.execute(table)

def create_table_output(port):
    tables = [
           """CREATE TABLE IF NOT EXISTS
                output_table(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time_load INTEGER NOT NULL,
                    time_warm INTEGER NOT NULL,
                    time_c INTEGER NOT NULL,
                    elapsed INTEGER NOT NULL,
                    throughput INTEGER NOT NULL,
                    power_cons INTEGER NOT NULL,
                    cpu_percent INTEGER NOT NULL,
                    gpu_percent INTEGER NOT NULL,
                    mem_percent INTEGER NOT NULL
                    )"""
        ]
    
    db = get_db(port)
    cursor = db.cursor()
    
    for table in tables:
        cursor.execute(table)

def delete_all_api(port):
    db = get_db(port)
    cursor = db.cursor()
    # DELETE
    queryread = "DELETE FROM readkey_table"
    cursor.execute(queryread)
    querywrite = "DELETE FROM writekey_table"
    cursor.execute(querywrite)
    db.commit()

def delete_all_cfg(port):
    db = get_db(port)
    cursor = db.cursor()
    # DELETE
    query = "DELETE FROM cfg_table"
    cursor.execute(query)
    db.commit()

def delete_all_output(port):
    db = get_db(port)
    cursor = db.cursor()
    # DELETE
    query = "DELETE FROM output_table"
    cursor.execute(query)
    db.commit()
