from firebase import firebase
def readFirebase():
    firebase1 = firebase.FirebaseApplication('https://cattleactivity-default-rtdb.firebaseio.com/', None)
    x = firebase1.get('/x', None)
    y = firebase1.get('/y', None)
    z = firebase1.get('/z', None)
    print(x,y,z)
    return(x,y,z)

#readFirebase()

def updateFirebase():
    firebase1 = firebase.FirebaseApplication('https://nmims-e780c-default-rtdb.firebaseio.com', None)
    firebase1.put('/nmims', 'medicine', 1)


#updateFirebase()