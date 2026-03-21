
def add_user(user_id: str, user_name: str):
    from .connect import conn
    select_querry = f"SELECT user_id FROM USERS WHERE user_id = {user_id}"
    with conn.cursor() as cur:
        cur.execute(select_querry)
        user = cur.fetchall()
    if len(user)==0:
        insert_querry = f"INSERT INTO USERS (user_id, user_name) VALUES ({user_id}, \'{user_name}\')"
        with conn.cursor() as cur:
            cur.execute(insert_querry)
            conn.commit()
        print(f"user {user_name} with id: {user_id} added!")
    else:
        print(f"user {user_name} with id: {user_id} is already exist")


    