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
