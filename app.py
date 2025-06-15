from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import re
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

load_dotenv()

app = Flask(__name__)
CORS(app)

# Gemini AI Setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# Walmart Products Database
WALMART_PRODUCTS = [
    {
        "id": 1001,
        "name": "iPhone 15 Pro",
        "features": "6.1-inch display, A17 Pro chip, 128GB, Pro camera system",
        "price": 99900,
        "offer": "Save $200 + Free shipping",
        "category": "Electronics",
        "brand": "Apple",
        "rating": 4.8,
        "in_stock": True
    },
    {
        "id": 1002,
        "name": "Samsung 65-inch 4K Smart TV",
        "features": "Crystal UHD, HDR10+, Tizen OS, Voice Remote",
        "price": 64900,
        "offer": "Save $300 + Free installation",
        "category": "Electronics",
        "brand": "Samsung",
        "rating": 4.6,
        "in_stock": True
    },
    {
        "id": 1003,
        "name": "HP Pavilion Gaming Laptop",
        "features": "15.6-inch FHD, Intel i7, 16GB RAM, 512GB SSD, RTX 4060",
        "price": 79900,
        "offer": "Save $400 - Rollback Price",
        "category": "Electronics",
        "brand": "HP",
        "rating": 4.5,
        "in_stock": True
    },
    {
        "id": 1004,
        "name": "KitchenAid Stand Mixer",
        "features": "5-quart bowl, 10-speed, tilt-head design, multiple attachments",
        "price": 29900,
        "offer": "Save $100 + Free attachments",
        "category": "Home & Kitchen",
        "brand": "KitchenAid",
        "rating": 4.9,
        "in_stock": True
    },
    {
        "id": 1005,
        "name": "Nike Air Max 270",
        "features": "Max Air unit, mesh upper, comfortable fit, multiple colors",
        "price": 12900,
        "offer": "Buy 2 Get 1 Free",
        "category": "Fashion",
        "brand": "Nike",
        "rating": 4.4,
        "in_stock": True
    },
    {
        "id": 1006,
        "name": "Instant Pot Duo Plus",
        "features": "9-in-1 pressure cooker, 6-quart, smart programs",
        "price": 9900,
        "offer": "Rollback - Save $40",
        "category": "Home & Kitchen",
        "brand": "Instant Pot",
        "rating": 4.7,
        "in_stock": True
    },
    {
        "id": 1007,
        "name": "Dyson V15 Detect Vacuum",
        "features": "Laser dust detection, powerful suction, up to 60min runtime",
        "price": 74900,
        "offer": "Save $200 + Free tool kit",
        "category": "Home & Kitchen",
        "brand": "Dyson",
        "rating": 4.6,
        "in_stock": True
    },
    {
        "id": 1008,
        "name": "PlayStation 5",
        "features": "4K gaming, SSD storage, DualSense controller, exclusive games",
        "price": 49900,
        "offer": "Bundle with 2 games - Save $100",
        "category": "Electronics",
        "brand": "Sony",
        "rating": 4.8,
        "in_stock": True
    },
    {
        "id": 1009,
        "name": "Levi's 501 Original Jeans",
        "features": "Classic straight fit, 100% cotton, multiple washes available",
        "price": 5900,
        "offer": "Buy 2 Get 30% off",
        "category": "Fashion",
        "brand": "Levi's",
        "rating": 4.3,
        "in_stock": True
    },
    {
        "id": 1010,
        "name": "Ninja Foodi Air Fryer",
        "features": "8-quart capacity, 8-in-1 functionality, digital display",
        "price": 19900,
        "offer": "Save $50 + Free recipe book",
        "category": "Home & Kitchen",
        "brand": "Ninja",
        "rating": 4.5,
        "in_stock": True
    },
    {
        "id": 1011,
        "name": "Olay Vitamin C Face Wash",
        "features": "Brightening formula, Vitamin C + Niacinamide, gentle daily cleanser",
        "price": 899,
        "offer": "Buy 2 Get 1 Free",
        "category": "Beauty & Personal Care",
        "brand": "Olay",
        "rating": 4.4,
        "in_stock": True
    },
    {
        "id": 1012,
        "name": "Neutrogena Vitamin C Gel Cleanser",
        "features": "Oil-free, Vitamin C infused, removes impurities, brightens skin",
        "price": 1299,
        "offer": "Save 25% + Free shipping",
        "category": "Beauty & Personal Care",
        "brand": "Neutrogena",
        "rating": 4.6,
        "in_stock": True
    },
    {
        "id": 1013,
        "name": "CeraVe Vitamin C Foaming Cleanser",
        "features": "Vitamin C + Hyaluronic Acid, fragrance-free, suitable for sensitive skin",
        "price": 1599,
        "offer": "Rollback - Save $200",
        "category": "Beauty & Personal Care",
        "brand": "CeraVe",
        "rating": 4.7,
        "in_stock": True
    },
    {
        "id": 1014,
        "name": "L'Oreal Paris Vitamin C Brightening Wash",
        "features": "10% Pure Vitamin C, anti-aging, removes makeup, dermatologist tested",
        "price": 799,
        "offer": "Special Price - Save 30%",
        "category": "Beauty & Personal Care",
        "brand": "L'Oreal",
        "rating": 4.3,
        "in_stock": True
    },
    {
        "id": 1015,
        "name": "The Ordinary Vitamin C Cleanser",
        "features": "Ascorbic Acid 8%, gentle exfoliation, brightening, vegan formula",
        "price": 699,
        "offer": "Limited Time - 40% off",
        "category": "Beauty & Personal Care",
        "brand": "The Ordinary",
        "rating": 4.5,
        "in_stock": True
    }
]

# High-priority clearance items
CLEARANCE_IDS = [1003, 1006, 1009, 1013, 1015]

def extract_json_response(text):
    """Extract JSON from AI response"""
    text = re.sub(r'```json\s*', '', text).strip()
    text = re.sub(r'```\s*$', '', text).strip()
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None

def create_backup_recommendation(products, query, budget):
    """Fallback recommendation logic"""
    keywords = query.lower().split()
    
    # Category matching
    categories = {
        'electronics': ['phone', 'laptop', 'tv', 'gaming', 'playstation', 'iphone', 'samsung'],
        'home & kitchen': ['kitchen', 'cooking', 'mixer', 'vacuum', 'instant pot', 'air fryer'],
        'fashion': ['clothes', 'jeans', 'shoes', 'nike', 'fashion', 'wear'],
        'beauty & personal care': ['facewash', 'cleanser', 'skincare', 'beauty', 'vitamin c', 'face wash', 'skin']
    }
    
    target_category = None
    for cat, cat_words in categories.items():
        if any(word in keywords for word in cat_words):
            target_category = cat
            break
    
    # Filter products
    filtered = products
    if target_category:
        filtered = [p for p in products if p['category'].lower() == target_category]
    
    # Sort by priority: clearance items first, then by rating and price
    def priority_score(product):
        score = 0
        if product['id'] in CLEARANCE_IDS:
            score += 1000
        if budget and product['price'] <= budget:
            score += 500
        score += product['rating'] * 100
        score -= product['price'] / 1000
        return score
    
    filtered.sort(key=priority_score, reverse=True)
    selected = filtered[:3]
    
    return {
        "product_ids": [p['id'] for p in selected],
        "summary": f"Top recommendations for '{query}' - featuring high-rated products with great deals"
    }

def log_interaction(user_id, query, response):
    """Log user interaction"""
    log_data = {
        "user_id": user_id,
        "query": query,
        "recommended_products": response.get("product_ids", []),
        "timestamp": datetime.now().isoformat()
    }
    print(f"📊 User interaction logged: {log_data}")
    # TODO: Save to actual database

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Product recommendation endpoint"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        query = data.get('query', '').strip()
        budget = data.get('budget')
        
        if not user_id or not query:
            return jsonify({"error": "user_id and query are required"}), 400

        # Filter available products
        available = [p for p in WALMART_PRODUCTS if p["in_stock"]]
        if budget:
            available = [p for p in available if p["price"] <= budget]
        
        if not available:
            return jsonify({
                "product_ids": [],
                "summary": "No products found within your criteria"
            })

        # Build AI context
        context = "\n".join([
            f"ID: {p['id']}, Name: {p['name']}, Brand: {p['brand']}, "
            f"Price: ₹{p['price']:,}, Rating: {p['rating']}/5, "
            f"Features: {p['features']}, Offer: {p['offer']}, "
            f"Category: {p['category']}" + 
            (" [CLEARANCE - PRIORITY]" if p['id'] in CLEARANCE_IDS else "")
            for p in available
        ])

        # AI prompt
        prompt = f"""You are Walmart's AI shopping assistant. Recommend products based on user needs.

User Query: "{query}"
{f"Budget: ₹{budget:,}" if budget else "No budget specified"}

Available Products:
{context}

IMPORTANT: Respond with ONLY valid JSON in this format:
{{
    "product_ids": [1001, 1002, 1003],
    "summary": "Why these products are perfect for your needs"
}}

Rules:
- Recommend 2-4 best matching products
- Prioritize CLEARANCE items if relevant
- Consider price, rating, and features
- Product IDs must be integers"""

        try:
            # Get AI recommendation
            response = llm.invoke([HumanMessage(content=prompt)])
            parsed = extract_json_response(response.content)

            if parsed and "product_ids" in parsed and "summary" in parsed:
                valid_ids = [p["id"] for p in available]
                
                # Ensure valid product IDs
                if isinstance(parsed["product_ids"][0], str):
                    parsed["product_ids"] = [int(re.sub(r'\D', '', str(pid))) 
                                           for pid in parsed["product_ids"]]
                
                parsed["product_ids"] = [pid for pid in parsed["product_ids"] 
                                       if pid in valid_ids]
                
                if parsed["product_ids"]:
                    log_interaction(user_id, query, parsed)
                    return jsonify(parsed)
                    
        except Exception as e:
            print(f"AI failed: {e}")
        
        # Use backup recommendation
        backup = create_backup_recommendation(available, query, budget)
        log_interaction(user_id, query, backup)
        return jsonify(backup)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500



if __name__ == '__main__':
    print("🛒 Walmart Recommendation API Starting...")
    print(f"📦 {len(WALMART_PRODUCTS)} products loaded")
    print(f"🏷️ {len(CLEARANCE_IDS)} clearance items available")
    print("🌐 Single API Endpoint: POST /recommend")
    
    app.run(debug=True, host='0.0.0.0', port=5000)