import numpy as np
from django.shortcuts import render

# Create your views here.

from rest_framework.response import Response
from rest_framework.views import APIView


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

class DiagnoseView(APIView):
    def post(self, request):
        data_input = request.data.get('instance')
        print('This is for diagnosis...' + data_input)
        diagnose = self.predict_diagnose(data_input)
        return Response({"diagnose": diagnose})

    def predict_diagnose(self, data_input):
        X_train = np.array([
                    "11/7 3	8 17 2 no ceasarian section	yes	yes	died foetus	6.5	2 clear	12 palm itching feet itching unable to sleep",
                    "10/7 3	8 17 0 no vaginal no no 7 clear	13.2",
                    "11/8 3 9 11 2 no vaginal yes yes 6.7 clear 11.4 palm itching feet itching",
                    "12/10 3 8 12 3	no vaginal no no nausea 7 clear 12.5 intractable nausea vomitting dehydration",
                    "13/8 3 7 12 1 yes no no 7.5 mild proteinuria 14.2 glycaemic status",
                    "11/8 3	7 10 2 no vaginal no no	6.9	clear 12.7 fatigue",
                    "12/7 3	9 13 1 no yes yes 6.2 clear	13 abdominal pain",
                    "10/7 3 8 10 1 no no yes 5.8 clear 12.8	minor dysuria symptoms and pain of a left side renal colic character",
                    "14/10 3 7 10 2	no ceasarian section yes yes 7.3 mild=proteinuria 8.3 high blood pressure",
                    "13/6 3	7 10 2 no vaginal no no	5.5	clear 3.7 abdominal pain vomitting",
                    "12/7 3	8 10 1 no no no	6 3+ of protein	12.9 heavy menstruation",
                    "10/8 3	9 9	2 no vaginal no	no 6.2 clear 9.5 missed period",
                    "12/8 3	8 10 1 no no yes 8 clear 11	intractable vomiting",
                    "10/7 3	8 8 no no no 5.7 clear 9.3 lower limb pain",
                    "9/5 3 7 11	2 no vaginal no	no 5.9 clear 8 high fever",
                    "14/9 3	7 10 1 no no no 6.1 clear 10.3",
                    "8/5 3 7 12	5 no vaginal no	no 6.3 clear 7.9 abdominal pain vaginal bleeding",
                    "12/7 3	8 8	2 yes vaginal no no trace glucose 13.2 irregular period",
                    "11/8 3	9 12 12	yes	both yes yes 6.2 urinary tract infection 10.5 missed period",
                    "17/10 3 9 13 2	yes	ceasarian section yes yes 7 urinary tract infection	8.5	body itching",
                    "11/8 3	8 7 2 no vaginal no no 6.8 clear 11.6 hypertension weakness",
                    "13/8 3 7 8 2 yes vaginal no no 6.4 mild proteinuria 10.7 hypertension",
                    "13/8 3 6 8 1 no no no 7.8 clear 12.6 edema weakness",
                    "12/7 3 7 10 1 no no no 7.3 clear 7.4 hypertension",
                    "16/10 3 8 10 1 no no no 5.9 clear 11 hypertension dorsocervical fat facial hirsutism edema",
                    "9/6 3 8 11 1 no no no 6.6 clear 12.2 drowsiness respiratory distress cardiac failure",
                    "10/8 3 9 8 1 yes no no clear 11.6 eyes discolouration nailbuds and face body itching sleepless",
                    "11/7 3	8 7 1 no no no 8 clear 12.4 palm itching intrahepatic cholestasis of pregnancy"
                    ])
        y_train_text = [["intrahepatic cholestasis of pregnancy"], 
                        ['palpitations and episodic lightheadedness'],
                        ['intrahepatic cholestasis of pregnancy'],
                        ["elevated aminotransferases"],
                        ["gestational diabetes mellitus"],
                        ["diabetic ketoacidosis (DKA) with adult-onset type 1 diabetes mellitus"],
                        ["placenta previa percreta"],
                        ["fetal urosepsis"],
                        ["phaeochromocytoma complicating pregnancy"],
                        ["vitamin b12 deficiency anaemia"],
                        ["severe preeclampsia"],
                        ["gestational diabetes mellitus"],
                        ["hyperemesis gravidarum"],
                        ["ovarian vein thrombosis"],
                        ["ovarian vein thrombosis"],
                        ["parvovirus infection"],
                        ["placenta previa percreta"],
                        ["type2 diabetes"],
                        ["type2 diabetes"],
                        ["intrahepatic cholestasis of pregnancy"],
                        ["cushing's syndrome"],
                        ["cushing's syndrome"],
                        ["cushing's syndrome"],
                        ["cushing's syndrome"],
                        ["cushing's syndrome"],
                        ["peripartum cardiomyopathy"],
                        ["intrahepatic cholestasis of pregnancy"],
                        ["intrahepatic cholestasis of pregnancy"]
                        ]

        X_test = np.array([data_input])
        target_names = ['intrahepatic cholestasis of pregnancy', 'palpitations and episodic lightheadedness']

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(y_train_text)

        classifier = Pipeline([('vectorizer', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', OneVsRestClassifier(LinearSVC()))])

        classifier.fit(X_train, Y)
        predicted = classifier.predict(X_test)
        all_labels = mlb.inverse_transform(predicted)
        return all_labels

class TreatmentView(APIView):
    def post(self, request):
        data_input = request.data.get('instance')
        print('This is for treatment...' + data_input)
        treatment = self.predict_treatment(data_input)
        return Response({"treatment": treatment});

    def predict_treatment(self, data_input):
        X_train = np.array(["13/14 3 7 10 2 no vaginal no no abortion 2years ago 6.9 1.8 trace glucose 10.5	edema weakness hypertension",
                    "11/7 3	8 17 2 no ceasarian section	yes	yes	died foetus	6.5	2 clear	12 palm itching feet itching unable to sleep",
                    "10/7 3	8 17 0 no vaginal no no 7 clear	13.2",
                    "11/8 3 9 11 2 no vaginal yes yes 6.7 clear 11.4 palm itching feet itching",
                    "12/10 3 8 12 3	no vaginal no no nausea 7 clear 12.5 intractable nausea vomitting dehydration",
                    "13/8 3 7 12 1 yes no no 7.5 mild proteinuria 14.2 glycaemic status",
                    "11/8 3	7 10 2 no vaginal no no	6.9	clear 12.7 fatigue",
                    "12/7 3	9 13 1 no yes yes 6.2 clear 13 abdominal pain",
                    "10/7 3 8 10 1 no no yes 5.8 clear 12.8	minor dysuria symptoms and pain of a left side renal colic character",
                    "14/10 3 7 10 2	no ceasarian section yes yes 7.3 mild=proteinuria 8.3 high blood pressure",
                    "13/6 3	7 10 2 no vaginal no no	5.5	clear 3.7 abdominal pain vomitting",
                    "10/8 3	9 9	2 no vaginal no	no 6.2 clear 9.5 missed period",
                    "12/8 3	8 10 1 no no yes 8 clear 11	intractable vomiting",
                    "10/7 3	8 8 no no no 5.7 clear 9.3 lower limb pain",
                    "9/5 3 7 11	2 no vaginal no	no 5.9 clear 8 high fever",
                    "8/5 3 7 12	5 no vaginal no	no 6.3 clear 7.9 abdominal pain vaginal bleeding",
                    "12/7 3	8 8	2 yes vaginal no no trace glucose 13.2 irregular period",
                    "11/8 3	9 12 12	yes	both yes yes 6.2 urinary tract infection 10.5 missed period",
                    "17/10 3 9 13 2	yes	ceasarian section yes yes 7 urinary tract infection	8.5	body itching",
                    "11/8 3	8 7 2 no vaginal no no 6.8 clear 11.6 hypertension weakness",
                    "13/8 3 7 8 2 yes vaginal no no 6.4 mild proteinuria 10.7 hypertension",
                    "13/8 3 6 8 1 no no no 7.8 clear 12.6 edema weakness",
                    "12/7 3 7 10 1 no no no 7.3 clear 7.4 hypertension",
                    "16/10 3 8 10 1 no no no 5.9 clear 11 hypertension dorsocervical fat facial hirsutism edema",
                    "9/6 3 8 11 1 no no no 6.6 clear 12.2 drowsiness respiratory distress cardiac failure",
                    "10/8 3 9 8 1 yes no no clear 11.6 eyes discolouration nailbuds and face body itching sleepless"
                    ])
        y_train_text = [["Surgical resection with misoprostol", "ketoconazole(150mg/twice daily)"],
                        ["ursodeoxycholic acid"],
                        ["ursodeoxycholic acid"],
                        ["antiemetics", "IV fluids"],
                        ["insulin therapy twice daily injections of human biphasic isophane insulin with follow ups 2-4weeks"],
                        ["salin and intensive insulin therapy"],
                        ["preventive catheterization of the descending aorta via transhumeral access", "stark cesarean delivery", 
                        "uterotonics drugs", "affronti endouterine square hemostatic sutures", "intrauterine application of barki balloon and partial filling with 100ml of normal saline",
                        "reversible radiological embolization and/or surgical ligation of the uterine arteries"],
                        ["cylastatyna + iipenem 0.5 G four times daily and clindamycin 0.3 G three times daily"],
                        ["30mg phenoxybenzamine in the morning and 10mg in the evening, propranolol after several days"],
                        ["10 units of blood and 5 vitamins b12 injection for 8days"],
                        ["iv antihypertensive drugs such as 25mg labetalol bolus repeat above 15minute intervals"],
                        ["salin and intensive insulin therapy"],
                        ["lower molecular weight heparin (LMWH, enoxaparin 4000IU 2ce daily",
                        "intavenous cephalosporin and compression stockings therapy with oral anticoagulant 2mg once daily"],
                        ["intravenous heparin for 5days and antibiotics"],
                        ["intrauterine blood transfusion"],
                        ["cylastatyna + iipenem 0.5 G four times daily"],
                        ["insulin therapy"],
                        ["cesarean section under general anesthesia"],
                        ["ursodeoxycholic acid"],
                        ["metyrapone 500mg daily", "hydrocortisone 25mg daily"],
                        ["metyrapone 500mg daily"],
                        ["hydrocortisone 30mg"],
                        ["hydrocortisone 30mg and surgical delivery"],
                        ["dobutamine infusion therapy", "oral digoxin therapy", "diuretics anticoagulation"],
                        ["ursodeoxycholic acid 300mg", "TID 2 pints 0f FFp", "pint of packed cell transfusion"],
                        ["ursodeoxycholic acid 250mg 3ce daily"]
                        ]

        X_test = np.array([data_input])
        target_names = ['Surgical resection with misoprostol', "ketoconazole(150mg/twice daily)", "ursodeoxycholic acid"]

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(y_train_text)

        classifier = Pipeline([('vectorizer', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', OneVsRestClassifier(LinearSVC()))])

        classifier.fit(X_train, Y)
        predicted = classifier.predict(X_test)
        all_labels = mlb.inverse_transform(predicted)
        return all_labels
