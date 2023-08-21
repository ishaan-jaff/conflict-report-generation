Objective: Convert the existing Python script into an API that can receive requests and return responses based on the provided upper and lower right watchbox coordinates, start date, and end date.

API Framework Options:
- Flask
- FastAPI

Server:
- Gunicorn

Sample Test of Existing Baseline Script:
```python
python generate_report.py --upper_left 12.63 -77.96 --lower_right 1.24 -73.9 --dates 20230614 20230714
```

Sample upper/right coordinates:
Name	Coordinates
Panama Gap	9.494792, -80.001465, 8.882720, -79.454392
Panama	10.576151, -83.028792, 6.732025, -77.085188
Western Colombia	12.633256, -77.958936, 1.245086, -73.937940
Venezeula	12.633256, -73.608350, 6.504604, -59.172315
Horn of Africa	14.997600, 35.511934, -2.485471, 55.435562
Sudan	22.455949, 21.674007, 9.084994, 38.724788
Buenaventura	4.627230, -78.267673, 3.147530, -76.004489
