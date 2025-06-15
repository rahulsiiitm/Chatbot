import requests
import json

# API Base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("🏥 Health Check:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def test_get_products():
    """Test get all products"""
    response = requests.get(f"{BASE_URL}/products")
    print("📦 All Products:")
    data = response.json()
    print(f"Total Products: {data['total_products']}")
    for product in data['products'][:3]:  # Show first 3
        print(f"  {product['id']}: {product['name']} - ₹{product['price']:,}")
    print("-" * 50)

def test_get_deadstock():
    """Test deadstock information"""
    response = requests.get(f"{BASE_URL}/deadstock")
    print("📋 Deadstock Information:")
    data = response.json()
    print(f"Deadstock IDs: {data['deadstock_config']['deadstock_ids']}")
    print("Deadstock Products:")
    for product in data['deadstock_products']:
        print(f"  {product['id']}: {product['name']} - {product['offer']}")
    print("-" * 50)

def test_recommendation(query, budget=None):
    """Test recommendation endpoint"""
    payload = {"message": query}
    if budget:
        payload["budget"] = budget
    
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    
    print(f"🤖 Recommendation for: '{query}'")
    if budget:
        print(f"💰 Budget: ₹{budget:,}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"📋 Summary: {data['summary']}")
        print(f"🎯 Recommended Products: {data['product_ids']}")
        
        if 'deadstock_promoted' in data and data['deadstock_promoted']:
            print(f"📦 Deadstock Promoted: {data['deadstock_promoted']}")
        
        if 'recommended_products' in data:
            print("📱 Product Details:")
            for product in data['recommended_products']:
                deadstock_marker = " [DEADSTOCK]" if product['id'] in data.get('deadstock_promoted', []) else ""
                print(f"  • {product['name']}{deadstock_marker}")
                print(f"    Price: ₹{product['price']:,} | Offer: {product['offer']}")
                print(f"    Features: {product['features'][:80]}...")
                print()
    else:
        print(f"❌ Error: {response.json()}")
    
    print("-" * 50)

def main():
    print("🚀 Testing Product Recommendation API")
    print("=" * 60)
    
    try:
        # Test basic endpoints
        test_health()
        test_get_products()
        test_get_deadstock()
        
        # Test various recommendation scenarios
        test_cases = [
            ("I need a budget smartphone under 20000", 20000),
            ("Looking for a gaming laptop with good graphics", None),
            ("Best noise cancelling headphones", None),
            ("Need a business laptop for office work", 50000),
            ("Affordable wireless earbuds", 5000),
            ("High-end smartphone with best camera", None)
        ]
        
        for query, budget in test_cases:
            test_recommendation(query, budget)
            
    except requests.ConnectionError:
        print("❌ Connection Error: Make sure the Flask API is running on http://localhost:5000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()