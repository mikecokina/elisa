import numpy as np
import json

qs = [68.27, 95.45, 99.73]


def main():
    # with open("inclination.json", "r") as f:
    with open("phase.json", "r") as f:
        data = f.read()
        data = json.loads(data)

    results = {}
    for i, _data in data.items():
        y = np.array(_data["y"])
        y = y[~np.isnan(y)]
        mn = np.mean(y)
        y -= mn
        y = np.abs(y)

        # results[i] = {q: "{:.6f}".format(val) for q, val in zip(qs, np.percentile(y, qs).tolist())}
        results[i] = {qs[0]: np.percentile(y, qs[0]),
                      'mean': mn}


    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
