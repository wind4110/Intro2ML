#!/usr/bin/python3

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import joblib

enron_data = joblib.load(open("final_project/final_project_dataset.pkl", "rb"))

num_persons = len(enron_data)
print(f"Number of persons in the dataset: {num_persons}")

num_features = len(next(iter(enron_data.values())))
print(f"Number of features for each person: {num_features}")

num_poi = sum(1 for person in enron_data.values() if person.get("poi") == True)
print(f"Number of POIs in the dataset: {num_poi}")

POI_names = []
with open("final_project/poi_names.txt", "r") as poi_file:
    for line in poi_file:
        line = line.strip()
        if line.startswith("(y)") or line.startswith("(n)"):
            name = line[4:]
            POI_names.append(name)

print(f"Number of POI names in the file: {len(POI_names)}")

print("\nTotal stock value for James Prentice:",
      enron_data["PRENTICE JAMES"]["total_stock_value"])

print("Number of email messages from Wesley Colwell to POIs:",
      enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

print("Value of stock options exercised by Jeffrey Skilling:",
      enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

print("Total payments to Kenneth Lay:",
      enron_data["LAY KENNETH L"]["total_payments"])
print("Total payments to Jeffrey Skilling:",
      enron_data["SKILLING JEFFREY K"]["total_payments"])
print("Total payments to Andrew Fastow:",
      enron_data["FASTOW ANDREW S"]["total_payments"])
print("The person taking home the most money",
      max(("LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S"),
          key=lambda name: enron_data[name]["total_payments"]))

num_quantified_salary = sum(1 for person in enron_data.values()
                            if person.get("salary") != "NaN")
print(f"\nNumber of people with quantified salary: {num_quantified_salary}")

num_known_email = sum(1 for person in enron_data.values()
                      if person.get("email_address") != "NaN")
print(f"Number of people with known email address: {num_known_email}")

# Ensure parent directory is in sys.path so 'tools' can be imported
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.feature_format import featureFormat, targetFeatureSplit

num_missing_total_payments = sum(1 for person in enron_data.values()
                                 if person.get("total_payments") == "NaN")
print(f"Number of people with 'NaN' for total payments: {num_missing_total_payments}")
print(f"Percentage of people with 'NaN' for total payments: "
      f"{(num_missing_total_payments / num_persons) * 100:.2f}%")

num_missing_total_payments_poi = sum(1 for person in enron_data.values()
                                     if person.get("poi") == True and person.get("total_payments") == "NaN")
print(f"Number of POIs with 'NaN' for total payments: {num_missing_total_payments_poi}")
print(f"Percentage of POIs with 'NaN' for total payments: "
      f"{(num_missing_total_payments_poi / num_poi) * 100:.2f}%")


