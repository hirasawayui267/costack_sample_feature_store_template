import json
import numpy as np

"""
Supply a 2D array, normalize each column and return the normalized 2D array
Sample request event:
{
    data:[
        [1, 2, 3],
        [4, 5, 6]
    ]
}
Response body:
{
    data: [
        [0.24253562503633297, 0.3713906763541037, 0.4472135954999579],
        [0.9701425001453319, 0.9284766908852594, 0.8944271909999159]
    ]
}
"""
def normalize_columns(event, context):
    # the input event is in a json string
    if type(event)==str:
        event = json.loads(event)
    matrix = np.array(event.get("data", [[]]))
    norm = np.linalg.norm(matrix, axis = 0)
    normalized_matrix = matrix / norm.reshape(1, 3)
    return {"data": normalized_matrix.tolist() }

"""
Supply a 2D array with [x, y] pairs, returns the m, b for regression y = mX + b
Sample request event:
{
    data:[
        [1, 3],
        [2, 4],
        [5, 6],
        [7, 8]
    ]
}
Response body:
{
    "m": 4.5,
    "b": 5.7
}
"""
def linear_regression2D(event, context):
    # the input event is in a json string
    if type(event)==str:
        event = json.loads(event)
    matrix = np.array(event.get("data", [[]]))
    if (len(matrix.shape) != 2 or matrix.shape[1] != 2):
        return {"response": "Data must be a 2D array of x, y pairs" }
    mean = np.mean(matrix, axis = 0)
    
    standard_deviation = np.std(matrix, axis = 0)
    correlation = np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]
    print("correlation", correlation)
    m = standard_deviation[1] / standard_deviation[0] * correlation
    b = mean[1] - m * mean[0]

    return {
        "m": m,
        "b": b
    }
