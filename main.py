from os import system
def menu():
    while 1:
        system('cls')
        print("1.Face Recognition\n2.Finger Counting\n3.Gender Classification\n4.Emotion Recognition\n5.Admin Panel\n6.Exit")
        try:
            choice1=int(input("Enter your choice : "))
            if choice1 == 1:
                from FaceRecognition.faceRecognition import FRecognition
                FRecognition()

            elif choice1 == 2:
                from FingerCount.handRecognition import Hrecogniton
                Hrecogniton()

            elif choice1 == 3:
                from GenderClassification.genderClassification import Gclassification
                Gclassification()

            elif choice1 == 4:
                from EmotionRecognition.emotionRecognition import ERecognition
                ERecognition()

            elif choice1 == 5:
                system('cls')
                import getpass
                password = getpass.getpass()
                mode = 'e'
                key = int(getpass.getpass("Key: "))
                # key=int(input('key:'))
                from Extra.Cypher import getTranslatedMessage
                Cpassword = getTranslatedMessage(mode, password, key)
                if Cpassword == "qbthuqp":
                    while 1:
                        system('cls')
                        print("Welcome Sidharth!")
                        print("\n1.Add new Person(for Face Recognition)"
                              "\n2.Retrain Face Recognition model"
                              "\n3.Show names of Person(Recognized by this program)"
                              "\n4.Retrain Emotion Recognition model using KDEF database"
                              "\n5.Retrain Emotion Recognition model using self collected database"
                              "\n6.Show number of Emotions(Recognized by this program)"
                              "\n7.Retrain Hand Recognition Model"
                              "\n8.Retrain Gender Classification Model"
                              "\n9.Logout and Return to Main Menu")
                        choice2 = int(input("Enter your choice : "))
                        if choice2 == 1:
                            from FaceRecognition.Face_recognition_preprocessing import face_data
                            from FaceRecognition.Face_recognition_preprocessing import Ftrain
                            face_data()
                            Ftrain()
                        elif choice2 == 2:
                            from FaceRecognition.Face_recognition_preprocessing import Ftrain
                            Ftrain()
                        elif choice2 == 3:
                            import pandas as pd
                            data = pd.read_csv('FaceRecognition/name_dict.csv', delimiter=',', header=None)
                            data.columns = ['id', 'name']
                            names = list(data['name'])
                            print(names)
                            system("pause")
                        elif choice2 == 4:
                            from EmotionRecognition.emotion_recognition_preprocessing import Etrain
                            Etrain(flag=1)
                            system("pause")
                        elif choice2 == 5:
                            from EmotionRecognition.emotion_recognition_preprocessing import Etrain
                            Etrain(flag=2)
                            system("pause")
                        elif choice2 == 6:
                            import pandas as pd
                            data = pd.read_csv('EmotionRecognition/emotion_dict.csv', delimiter=',', header=None)
                            data.columns = ['id', 'name']
                            emotion = list(data['name'])
                            print(emotion)
                            system("pause")
                        elif choice2 == 7:
                            from FingerCount.Handpreprocessing import Htrain
                            Htrain()
                            system("pause")
                        elif choice2 == 8:
                            from GenderClassification.gender_classification_preprocessing import Gtrain
                            Gtrain()
                            system("pause")
                        elif choice2 == 9:
                            break
                else:
                    print("Wrong password!")
                    system("pause")

            elif choice1 == 6:
                break
            else:
                print("\nRetry\n")
                system("pause")

        except:
            print("\nRetry\n")
            system("pause")

menu()
