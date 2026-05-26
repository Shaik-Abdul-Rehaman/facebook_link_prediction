import requests
import json
import time

BASE_URL = 'http://localhost:5000'

def test_health():
    print("\n[TEST 1] Health Check")
    print("-" * 40)
    try:
        response = requests.get(f'{BASE_URL}/health')
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    print("\n[TEST 2] Model Info")
    print("-" * 40)
    try:
        response = requests.get(f'{BASE_URL}/api/v1/model-info')
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Model: {data.get('model')}")
        print(f"Version: {data.get('version')}")
        print(f"Max batch size: {data.get('max_batch_size')}")
        print(f"Available models: {data.get('available_models')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_prediction():
    print("\n[TEST 3] Single Edge Prediction")
    print("-" * 40)
    payload = {
        'edges': [
            {'source_node': 1, 'destination_node': 2},
            {'source_node': 3, 'destination_node': 4},
            {'source_node': 5, 'destination_node': 6}
        ]
    }
    try:
        response = requests.post(
            f'{BASE_URL}/api/v1/predict',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        if response.status_code == 200:
            print(f"Total predictions: {len(data['predictions'])}")
            for pred in data['predictions'][:3]:
                print(f"  {pred['source_node']} -> {pred['destination_node']}: "
                      f"Prediction={pred['prediction']}, Probability={pred['probability']:.4f}")
        else:
            print(f"Error: {data.get('error')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    print("\n[TEST 4] Batch Prediction (10 edges)")
    print("-" * 40)
    edges = [
        {'source_node': i, 'destination_node': i+100}
        for i in range(1, 11)
    ]
    payload = {'edges': edges}
    try:
        response = requests.post(
            f'{BASE_URL}/api/v1/batch-predict',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        if response.status_code == 200:
            print(f"Total predictions: {data['total']}")
            print(f"Sample predictions:")
            for pred in data['predictions'][:3]:
                print(f"  {pred['source_node']} -> {pred['destination_node']}: "
                      f"Prob={pred['probability']:.4f}")
        else:
            print(f"Error: {data.get('error')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_metrics():
    print("\n[TEST 5] Get Model Metrics")
    print("-" * 40)
    try:
        response = requests.get(f'{BASE_URL}/api/v1/metrics')
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            for model, metrics in data.get('metrics', {}).items():
                print(f"\n{model}:")
                for key, val in metrics.items():
                    print(f"  {key}: {val:.4f}")
        else:
            print(f"Error: {response.json().get('error')}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_large_batch():
    print("\n[TEST 6] Large Batch (1000 edges)")
    print("-" * 40)
    edges = [
        {'source_node': i, 'destination_node': (i * 13) % 1862220 + 1}
        for i in range(1, 1001)
    ]
    payload = {'edges': edges}
    try:
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/api/v1/batch-predict',
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        elapsed = time.time() - start
        print(f"Status: {response.status_code}")
        data = response.json()
        if response.status_code == 200:
            print(f"Predictions: {data['total']}")
            print(f"Time taken: {elapsed:.2f}s")
            print(f"Average per edge: {elapsed/data['total']*1000:.2f}ms")
        else:
            print(f"Error: {data.get('error')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 60)
    print("LINK PREDICTION API - TEST SUITE")
    print("=" * 60)

    results = {
        'Health Check': test_health(),
        'Model Info': test_model_info(),
        'Single Prediction': test_single_prediction(),
        'Batch Prediction': test_batch_prediction(),
        'Metrics': test_metrics(),
        'Large Batch': test_large_batch()
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)

if __name__ == '__main__':
    print("\nMake sure the API server is running: python run_api.py")
    input("Press Enter to start tests...")
    main()
