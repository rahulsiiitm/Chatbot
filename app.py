from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)
CORS(app)

class WalmartRecommendationSystem:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        
        # User memories
        self.user_memories = {}
        
        # Products database
        self.products = [
            {"id": 1001, "name": "iPhone 15 Pro", "features": "6.1-inch display, A17 Pro chip, 128GB, Pro camera system", "price": 99900, "offer": "Save $200 + Free shipping", "category": "Electronics", "brand": "Apple", "rating": 4.8, "in_stock": True},
            {"id": 1002, "name": "Samsung 65-inch 4K Smart TV", "features": "Crystal UHD, HDR10+, Tizen OS, Voice Remote", "price": 64900, "offer": "Save $300 + Free installation", "category": "Electronics", "brand": "Samsung", "rating": 4.6, "in_stock": True},
            {"id": 1003, "name": "HP Pavilion Gaming Laptop", "features": "15.6-inch FHD, Intel i7, 16GB RAM, 512GB SSD, RTX 4060", "price": 79900, "offer": "Save $400 - Rollback Price", "category": "Electronics", "brand": "HP", "rating": 4.5, "in_stock": True},
            {"id": 1004, "name": "KitchenAid Stand Mixer", "features": "5-quart bowl, 10-speed, tilt-head design, multiple attachments", "price": 29900, "offer": "Save $100 + Free attachments", "category": "Home & Kitchen", "brand": "KitchenAid", "rating": 4.9, "in_stock": True},
            {"id": 1005, "name": "Nike Air Max 270", "features": "Max Air unit, mesh upper, comfortable fit, multiple colors", "price": 12900, "offer": "Buy 2 Get 1 Free", "category": "Fashion", "brand": "Nike", "rating": 4.4, "in_stock": True},
            {"id": 1011, "name": "Olay Vitamin C Face Wash", "features": "Brightening formula, Vitamin C + Niacinamide, gentle daily cleanser", "price": 899, "offer": "Buy 2 Get 1 Free", "category": "Beauty & Personal Care", "brand": "Olay", "rating": 4.4, "in_stock": True},
            {"id": 1012, "name": "Neutrogena Vitamin C Gel Cleanser", "features": "Oil-free, Vitamin C infused, removes impurities, brightens skin", "price": 1299, "offer": "Save 25% + Free shipping", "category": "Beauty & Personal Care", "brand": "Neutrogena", "rating": 4.6, "in_stock": True},
            {"id": 1013, "name": "CeraVe Vitamin C Foaming Cleanser", "features": "Vitamin C + Hyaluronic Acid, fragrance-free, suitable for sensitive skin", "price": 1599, "offer": "Rollback - Save $200", "category": "Beauty & Personal Care", "brand": "CeraVe", "rating": 4.7, "in_stock": True},
            {"id": 1014, "name": "Dyson V15 Detect Vacuum", "features": "Laser dust detection, powerful suction, up to 60min runtime", "price": 74900, "offer": "Save $200 + Free tool kit", "category": "Home & Kitchen", "brand": "Dyson", "rating": 4.6, "in_stock": True},
            {"id": 1015, "name": "PlayStation 5", "features": "4K gaming, SSD storage, DualSense controller, exclusive games", "price": 49900, "offer": "Bundle with 2 games - Save $100", "category": "Electronics", "brand": "Sony", "rating": 4.8, "in_stock": True}
        ]
        
        self.setup_chain()
    
    def setup_chain(self):
        """Setup the recommendation chain with the same prompt"""
        recommendation_prompt = PromptTemplate(
            input_variables=["query", "retrieved_products", "user_history", "budget", "available_product_ids"],
            template="""
            You are Walmart's AI shopping assistant. Analyze the user query and recommend the BEST matching products from the available inventory.
            
            User Query: {query}
            Budget Constraint: {budget}
            User Shopping History: {user_history}
            Available Product IDs: {available_product_ids}
            
            Retrieved Product Information:
            {retrieved_products}
            
            CRITICAL INSTRUCTIONS:
            1. ONLY recommend products from the Available Product IDs list above
            2. Choose 2-4 products that BEST match the user's specific query
            3. Consider price/budget constraints strictly
            4. Prioritize products with better ratings and special offers
            5. Use user history to personalize recommendations
            6. Focus on exact query matches first, then similar products
            
            RESPONSE FORMAT (JSON ONLY - NO OTHER TEXT):
            {{
                "product_ids": [actual_product_ids_only],
                "summary": "Brief explanation of why these products match the query and budget",
                "personalized_note": "Note based on user history if available, otherwise omit this field"
            }}
            
            Remember: 
            - product_ids must be actual IDs from Available Product IDs list
            - Match the query intent precisely
            - Respect budget limits
            - Maximum 4 recommendations
            """
        )
        
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=recommendation_prompt)
    
    def get_user_memory(self, user_id: str):
        """Get user memory"""
        return self.user_memories.get(user_id, [])
    
    def update_user_memory(self, user_id: str, query: str, recommended_products: List[int]):
        """Update user's shopping history"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = []
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "recommended_products": recommended_products
        }
        
        self.user_memories[user_id].append(interaction)
        
        # Keep only last 10 interactions
        if len(self.user_memories[user_id]) > 10:
            self.user_memories[user_id].pop(0)
    
    def get_matching_products(self, query: str, budget: int = None) -> List[Dict]:
        """Get products matching the query and budget"""
        query_lower = query.lower()
        available_products = [p for p in self.products if p["in_stock"]]
        
        # Apply budget filter
        if budget:
            available_products = [p for p in available_products if p['price'] <= budget]
        
        if not available_products:
            return []
        
        # Score products based on query match
        scored_products = []
        keywords = query_lower.split()
        
        for product in available_products:
            score = 0
            product_text = f"{product['name']} {product['features']} {product['category']} {product['brand']}".lower()
            
            # Primary relevance check - must have strong keyword match
            primary_match = False
            
            # Check for exact phrase match in name or brand (highest priority)
            if query_lower in product['name'].lower() or query_lower in product['brand'].lower():
                score += 20
                primary_match = True
            
            # Check individual keywords in name/brand (high priority)
            name_brand_text = f"{product['name']} {product['brand']}".lower()
            keyword_matches = 0
            for keyword in keywords:
                if len(keyword) > 2:  # Skip very short words
                    if keyword in name_brand_text:
                        score += 8
                        keyword_matches += 1
                        primary_match = True
                    elif keyword in product_text:
                        score += 3
                        keyword_matches += 1
            
            # Require at least one strong keyword match for relevance
            if keyword_matches == 0:
                continue
                
            # Additional scoring only if we have primary relevance
            if primary_match:
                # Exact phrase match in full product text
                if query_lower in product_text:
                    score += 5
                
                # Bonus for multiple keyword matches
                if keyword_matches >= 2:
                    score += 5
                
                # Category relevance bonus
                for keyword in keywords:
                    if keyword in product['category'].lower():
                        score += 2
            
            # Only include products with meaningful relevance score
            if score >= 8:  # Minimum threshold for relevance
                # Rating bonus (small impact)
                score += product['rating'] * 0.5
                scored_products.append((product, score))
        
        # Sort by score and return top matches
        scored_products.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in scored_products[:8]]
    
    def extract_json_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from AI response"""
        text = text.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        json_patterns = [
            r'\{[^{}]*"product_ids"[^{}]*\}',
            r'\{.*?"product_ids".*?\}',
            r'\{.*\}',
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    parsed = json.loads(json_str)
                    if "product_ids" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def extract_budget_from_query(self, query: str) -> tuple[str, int]:
        """Extract budget from query"""
        budget_patterns = [
            r'under\s*₹?\s*(\d+(?:,\d+)*)',
            r'below\s*₹?\s*(\d+(?:,\d+)*)',
            r'within\s*₹?\s*(\d+(?:,\d+)*)',
            r'budget\s*of\s*₹?\s*(\d+(?:,\d+)*)',
            r'budget\s*₹?\s*(\d+(?:,\d+)*)',
            r'₹\s*(\d+(?:,\d+)*)',
            r'rs\.?\s*(\d+(?:,\d+)*)',
            r'less\s*than\s*₹?\s*(\d+(?:,\d+)*)',
            r'max\s*₹?\s*(\d+(?:,\d+)*)'
        ]
        
        budget = None
        cleaned_query = query
        
        for pattern in budget_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                budget_str = match.group(1).replace(',', '')
                try:
                    budget = int(budget_str)
                    cleaned_query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
                    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
                    break
                except ValueError:
                    continue
        
        return cleaned_query, budget

    def recommend_products(self, user_id: str, query: str) -> Dict[str, Any]:
        """Main recommendation function"""
        try:
            # Extract budget from query
            cleaned_query, budget = self.extract_budget_from_query(query)
            
            # Get matching products
            candidate_products = self.get_matching_products(cleaned_query, budget)
            
            if not candidate_products:
                return {
                    "product_ids": [],
                    "products": [],
                    "summary": f"Sorry, we don't have '{cleaned_query}' in stock" + (f" within your budget of ₹{budget:,}" if budget else "") + ". Please try a different search or check back later."
                }
            
            # Get user history
            user_history = self.get_user_memory(user_id)
            history_summary = "New user - no previous history"
            if user_history:
                recent_queries = [h.get('query', '') for h in user_history[-3:]]
                history_summary = f"Recent searches: {', '.join(recent_queries[-3:])}"
            
            # Prepare data for AI recommendation
            available_ids = [p['id'] for p in candidate_products]
            product_details = "\n".join([
                f"ID: {p['id']}, Name: {p['name']}, Price: ₹{p['price']:,}, Rating: {p['rating']}, Features: {p['features']}"
                for p in candidate_products
            ])
            
            # Get AI recommendation
            chain_input = {
                "query": cleaned_query,
                "retrieved_products": product_details,
                "user_history": history_summary,
                "budget": f"₹{budget:,}" if budget else "No budget limit",
                "available_product_ids": str(available_ids)
            }
            
            recommendation_result = self.recommendation_chain.run(**chain_input)
            parsed = self.extract_json_response(recommendation_result)
            
            if parsed and "product_ids" in parsed and parsed["product_ids"]:
                valid_ids = [pid for pid in parsed["product_ids"] if pid in available_ids]
                
                if valid_ids:
                    # Update user memory
                    self.update_user_memory(user_id, cleaned_query, valid_ids)
                    
                    # Get product details
                    recommended_products = [p for p in candidate_products if p['id'] in valid_ids]
                    
                    result = {
                        "product_ids": valid_ids,
                        "products": recommended_products,
                        "summary": parsed.get("summary", f"Found {len(valid_ids)} products matching your query")
                    }
                    
                    if "personalized_note" in parsed:
                        result["personalized_note"] = parsed["personalized_note"]
                    
                    return result
            
            # Fallback: return top products
            top_products = candidate_products[:3]
            self.update_user_memory(user_id, cleaned_query, [p['id'] for p in top_products])
            
            return {
                "product_ids": [p['id'] for p in top_products],
                "products": top_products,
                "summary": f"Top {len(top_products)} products matching '{cleaned_query}'" + (f" within budget ₹{budget:,}" if budget else "")
            }
            
        except Exception as e:
            print(f"Error: {e}")
            # Simple fallback
            cleaned_query, budget = self.extract_budget_from_query(query)
            products = self.get_matching_products(cleaned_query, budget)[:3]
            return {
                "product_ids": [p['id'] for p in products],
                "products": products,
                "summary": f"Found {len(products)} products for '{cleaned_query}'"
            }

# Initialize the system
walmart_ai = WalmartRecommendationSystem()

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Product recommendation endpoint"""
    try:
        data = request.get_json()
        user_id = data.get('id')  # Changed from 'user_id' to 'id'
        query = data.get('query', '').strip()
        
        if not user_id or not query:
            return jsonify({"error": "id and query are required"}), 400
        
        # Get recommendations
        result = walmart_ai.recommend_products(user_id, query)
        return jsonify(result)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_products": len(walmart_ai.products)
    })

if __name__ == '__main__':
    print("🚀 Walmart AI Recommendation System Starting...")
    print("📊 Endpoints:")
    print("   - POST /recommend - Get product recommendations (requires: id, query)")
    print("   - GET /health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)