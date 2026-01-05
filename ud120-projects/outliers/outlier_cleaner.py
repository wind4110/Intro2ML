#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    errors = abs(predictions - net_worths)
    data = list(zip(ages.flatten(), net_worths.flatten(), errors.flatten()))
    data.sort(key=lambda x: x[2])  # Sort by error
    limit = int(len(data) * 0.9)  # Keep only 90%
    cleaned_data = data[:limit]

    return cleaned_data

