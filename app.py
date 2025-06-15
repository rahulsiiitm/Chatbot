from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any

# LangChain imports for advanced features
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseRetriever

load_dotenv()

app = Flask(__name__)
CORS(app)

# Advanced LangChain Setup
class WalmartRecommendationSystem:
    def __init__(self):
        # Initialize LLM and Embeddings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1  # Lower temperature for more consistent responses
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 interactions
            return_messages=True
        )
        
        # User-specific memories
        self.user_memories = {}
        
        # Initialize vector database and RAG
        self.setup_vector_database()
        self.setup_chains()
        self.setup_agents()
    
    def setup_vector_database(self):
        """Create FAISS vector database with product information"""
        # Walmart Products Database
        self.products = [
            {
                "id": 1001, "name": "iPhone 15 Pro", "features": "6.1-inch display, A17 Pro chip, 128GB, Pro camera system",
                "price": 99900, "offer": "Save $200 + Free shipping", "category": "Electronics", "brand": "Apple", "rating": 4.8, "in_stock": True
            },
            {
                "id": 1002, "name": "Samsung 65-inch 4K Smart TV", "features": "Crystal UHD, HDR10+, Tizen OS, Voice Remote",
                "price": 64900, "offer": "Save $300 + Free installation", "category": "Electronics", "brand": "Samsung", "rating": 4.6, "in_stock": True
            },
            {
                "id": 1003, "name": "HP Pavilion Gaming Laptop", "features": "15.6-inch FHD, Intel i7, 16GB RAM, 512GB SSD, RTX 4060",
                "price": 79900, "offer": "Save $400 - Rollback Price", "category": "Electronics", "brand": "HP", "rating": 4.5, "in_stock": True
            },
            {
                "id": 1004, "name": "KitchenAid Stand Mixer", "features": "5-quart bowl, 10-speed, tilt-head design, multiple attachments",
                "price": 29900, "offer": "Save $100 + Free attachments", "category": "Home & Kitchen", "brand": "KitchenAid", "rating": 4.9, "in_stock": True
            },
            {
                "id": 1005, "name": "Nike Air Max 270", "features": "Max Air unit, mesh upper, comfortable fit, multiple colors",
                "price": 12900, "offer": "Buy 2 Get 1 Free", "category": "Fashion", "brand": "Nike", "rating": 4.4, "in_stock": True
            },
            {
                "id": 1011, "name": "Olay Vitamin C Face Wash", "features": "Brightening formula, Vitamin C + Niacinamide, gentle daily cleanser",
                "price": 899, "offer": "Buy 2 Get 1 Free", "category": "Beauty & Personal Care", "brand": "Olay", "rating": 4.4, "in_stock": True
            },
            {
                "id": 1012, "name": "Neutrogena Vitamin C Gel Cleanser", "features": "Oil-free, Vitamin C infused, removes impurities, brightens skin",
                "price": 1299, "offer": "Save 25% + Free shipping", "category": "Beauty & Personal Care", "brand": "Neutrogena", "rating": 4.6, "in_stock": True
            },
            {
                "id": 1013, "name": "CeraVe Vitamin C Foaming Cleanser", "features": "Vitamin C + Hyaluronic Acid, fragrance-free, suitable for sensitive skin",
                "price": 1599, "offer": "Rollback - Save $200", "category": "Beauty & Personal Care", "brand": "CeraVe", "rating": 4.7, "in_stock": True
            },
            {
                "id": 1014, "name": "Dyson V15 Detect Vacuum", "features": "Laser dust detection, powerful suction, up to 60min runtime",
                "price": 74900, "offer": "Save $200 + Free tool kit", "category": "Home & Kitchen", "brand": "Dyson", "rating": 4.6, "in_stock": True
            },
            {
                "id": 1015, "name": "PlayStation 5", "features": "4K gaming, SSD storage, DualSense controller, exclusive games",
                "price": 49900, "offer": "Bundle with 2 games - Save $100", "category": "Electronics", "brand": "Sony", "rating": 4.8, "in_stock": True
            }
        ]
        
        # Create documents for vector storage
        documents = []
        for product in self.products:
            content = f"""
            Product: {product['name']}
            Brand: {product['brand']}
            Category: {product['category']}
            Price: ₹{product['price']:,}
            Rating: {product['rating']}/5
            Features: {product['features']}
            Special Offer: {product['offer']}
            Product ID: {product['id']}
            """
            
            doc = Document(
                page_content=content,
                metadata={
                    "product_id": product['id'],
                    "name": product['name'],
                    "category": product['category'],
                    "price": product['price'],
                    "brand": product['brand']
                }
            )
            documents.append(doc)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Create retriever for RAG
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Retrieve more products for better selection
        )
    
    def setup_chains(self):
        """Setup LangChain chains"""
        # RAG Chain for product information retrieval
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        
        # Enhanced Recommendation Chain with better instructions
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
        
        self.recommendation_chain = LLMChain(
            llm=self.llm,
            prompt=recommendation_prompt
        )
    
    def setup_agents(self):
        """Setup LangChain agents with tools"""
        
        def search_products_by_category(category: str) -> str:
            """Tool to search products by category"""
            filtered = [p for p in self.products if p['category'].lower() == category.lower()]
            return json.dumps(filtered[:5])
        
        def search_products_by_price_range(price_input: str) -> str:
            """Tool to search products by price range"""
            try:
                cleaned_input = price_input.strip().strip('"\'')
                
                if ',' in cleaned_input:
                    parts = cleaned_input.split(',')
                    min_price = int(parts[0].strip())
                    max_price = int(parts[1].strip())
                else:
                    max_price = int(cleaned_input)
                    min_price = 0
                
                filtered = [p for p in self.products if min_price <= p['price'] <= max_price]
                return json.dumps(filtered[:5])
            except (ValueError, IndexError) as e:
                return json.dumps({"error": f"Invalid price format. Error: {str(e)}"})
        
        def get_product_details(product_id: str) -> str:
            """Tool to get detailed product information"""
            try:
                pid = int(product_id.strip())
                product = next((p for p in self.products if p['id'] == pid), None)
                return json.dumps(product) if product else json.dumps({"error": "Product not found"})
            except ValueError:
                return json.dumps({"error": "Invalid product ID format"})
        
        def search_by_keyword(keyword: str) -> str:
            """Tool to search products by keyword"""
            keyword = keyword.lower().strip()
            matches = []
            
            for product in self.products:
                searchable_text = f"{product['name']} {product['features']} {product['category']} {product['brand']}".lower()
                if keyword in searchable_text:
                    matches.append(product)
            
            return json.dumps(matches[:5])
        
        # Define tools for the agent
        tools = [
            Tool(
                name="SearchByCategory",
                func=search_products_by_category,
                description="Search products by category. Available categories: Electronics, Fashion, Beauty & Personal Care, Home & Kitchen"
            ),
            Tool(
                name="SearchByPriceRange",
                func=search_products_by_price_range,
                description="Search products by price range. Input format: 'min_price,max_price' or just 'max_price'"
            ),
            Tool(
                name="GetProductDetails",
                func=get_product_details,
                description="Get detailed information about a specific product by ID"
            ),
            Tool(
                name="SearchByKeyword",
                func=search_by_keyword,
                description="Search products by keyword or feature"
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def get_user_memory(self, user_id: str):
        """Get or create user-specific memory"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = []
        return self.user_memories[user_id]
    
    def update_user_memory(self, user_id: str, query: str, recommended_products: List[int]):
        """Update user's shopping history"""
        user_history = self.get_user_memory(user_id)
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "recommended_products": recommended_products
        }
        
        # Add product details to history
        for product_id in recommended_products:
            product = next((p for p in self.products if p['id'] == product_id), None)
            if product:
                interaction.update({
                    "category": product['category'],
                    "brand": product['brand']
                })
                break
        
        user_history.append(interaction)
        
        # Keep only last 10 interactions
        if len(user_history) > 10:
            user_history.pop(0)
    
    def filter_products_by_budget(self, products: List[Dict], budget: int = None) -> List[Dict]:
        """Filter products by budget constraint"""
        if budget is None:
            return products
        return [p for p in products if p['price'] <= budget]
    
    def get_best_matching_products(self, query: str, budget: int = None) -> List[Dict]:
        """Get best matching products using multiple strategies"""
        query_lower = query.lower()
        available_products = [p for p in self.products if p["in_stock"]]
        
        # Apply budget filter first
        if budget:
            available_products = self.filter_products_by_budget(available_products, budget)
        
        if not available_products:
            return []
        
        # Strategy 1: Exact phrase matching
        exact_matches = []
        for product in available_products:
            searchable_text = f"{product['name']} {product['features']} {product['brand']}".lower()
            if query_lower in searchable_text:
                exact_matches.append(product)
        
        # Strategy 2: Keyword scoring
        keywords = query_lower.split()
        scored_products = []
        
        for product in available_products:
            score = 0
            product_text = f"{product['name']} {product['features']} {product['category']} {product['brand']}".lower()
            
            # Exact phrase gets highest score
            if query_lower in product_text:
                score += 10
            
            # Individual keyword matching
            for keyword in keywords:
                if keyword in product_text:
                    score += 2
                    # Bonus for name/brand matching
                    if keyword in product['name'].lower() or keyword in product['brand'].lower():
                        score += 3
            
            # Rating bonus
            score += product['rating']
            
            if score > 0:
                scored_products.append((product, score))
        
        # Sort by score
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Return best matches, prioritizing exact matches
        if exact_matches:
            # Sort exact matches by rating
            exact_matches.sort(key=lambda x: x['rating'], reverse=True)
            return exact_matches[:4]
        elif scored_products:
            return [p[0] for p in scored_products[:4]]
        else:
            # Fallback: return highest rated products in budget
            return sorted(available_products, key=lambda x: x['rating'], reverse=True)[:3]
    
    def recommend_products(self, user_id: str, query: str, budget: int = None) -> Dict[str, Any]:
        """Main recommendation function using enhanced LangChain features"""
        try:
            print(f"🔍 Processing query: '{query}' for user {user_id} with budget: {budget}")
            
            # 1. Get best matching products using multiple strategies
            candidate_products = self.get_best_matching_products(query, budget)
            
            if not candidate_products:
                return {
                    "product_ids": [],
                    "products": [],
                    "summary": f"No products found matching '{query}'" + (f" within budget ₹{budget:,}" if budget else ""),
                    "method": "no_matches"
                }
            
            # 2. Use RAG to get detailed product information
            rag_result = self.rag_chain({"query": query})
            retrieved_info = rag_result['result']
            
            # 3. Get user history for personalization
            user_history = self.get_user_memory(user_id)
            history_summary = "New user - no previous history"
            if user_history:
                recent_categories = [h.get('category', '') for h in user_history[-3:]]
                recent_brands = [h.get('brand', '') for h in user_history[-3:]]
                history_summary = f"Recent interests: categories {set(recent_categories)}, brands {set(recent_brands)}"
            
            # 4. Prepare data for recommendation chain
            available_ids = [p['id'] for p in candidate_products]
            product_details = "\n".join([
                f"ID: {p['id']}, Name: {p['name']}, Price: ₹{p['price']:,}, Rating: {p['rating']}, Features: {p['features']}"
                for p in candidate_products
            ])
            
            # 5. Use recommendation chain with proper context
            chain_input = {
                "query": query,
                "retrieved_products": product_details,
                "user_history": history_summary,
                "budget": f"₹{budget:,}" if budget else "No budget limit",
                "available_product_ids": str(available_ids)
            }
            
            print(f"🧠 Available product IDs: {available_ids}")
            
            recommendation_result = self.recommendation_chain.run(**chain_input)
            print(f"📋 AI Response: {recommendation_result}")
            
            # 6. Parse and validate the result
            parsed = self.extract_json_response(recommendation_result)
            
            if parsed and "product_ids" in parsed and parsed["product_ids"]:
                # Validate that recommended IDs are in available products
                valid_ids = [pid for pid in parsed["product_ids"] if pid in available_ids]
                
                if valid_ids:
                    # Update user memory
                    self.update_user_memory(user_id, query, valid_ids)
                    
                    # Add product details
                    recommended_products = [p for p in candidate_products if p['id'] in valid_ids]
                    
                    result = {
                        "product_ids": valid_ids,
                        "products": recommended_products,
                        "summary": parsed.get("summary", f"Found {len(valid_ids)} products matching your query"),
                        "method": "langchain_enhanced"
                    }
                    
                    if "personalized_note" in parsed:
                        result["personalized_note"] = parsed["personalized_note"]
                    
                    print(f"✅ Final recommendation: {valid_ids}")
                    return result
            
            # 7. Fallback: return top candidate products
            print("🔄 Using candidate products as fallback...")
            top_products = candidate_products[:3]
            return {
                "product_ids": [p['id'] for p in top_products],
                "products": top_products,
                "summary": f"Top {len(top_products)} products matching '{query}'" + (f" within budget ₹{budget:,}" if budget else ""),
                "method": "candidate_fallback"
            }
            
        except Exception as e:
            print(f"❌ Advanced recommendation failed: {e}")
            import traceback
            traceback.print_exc()
            return self.simple_fallback(query, budget)
    
    def extract_json_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from AI response with better parsing"""
        # Clean the text
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Try to find JSON in the response
        json_patterns = [
            r'\{[^{}]*"product_ids"[^{}]*\}',  # Simple JSON pattern
            r'\{.*?"product_ids".*?\}',         # More flexible pattern
            r'\{.*\}',                          # Any JSON-like structure
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    parsed = json.loads(json_str)
                    if "product_ids" in parsed:
                        print(f"✅ Successfully parsed JSON: {parsed}")
                        return parsed
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing failed for pattern {pattern}: {e}")
                    continue
        
        print(f"❌ Could not extract JSON from: {text}")
        return None
    
    def simple_fallback(self, query: str, budget: int = None) -> Dict[str, Any]:
        """Enhanced fallback with better matching"""
        products = self.get_best_matching_products(query, budget)
        
        if not products:
            # Return top-rated products in budget as last resort
            available = [p for p in self.products if p["in_stock"]]
            if budget:
                available = self.filter_products_by_budget(available, budget)
            products = sorted(available, key=lambda x: x['rating'], reverse=True)[:3]
        
        return {
            "product_ids": [p['id'] for p in products],
            "products": products,
            "summary": f"Found {len(products)} products for '{query}'" + (f" within budget ₹{budget:,}" if budget else ""),
            "method": "enhanced_fallback"
        }

# Initialize the advanced system
walmart_ai = WalmartRecommendationSystem()

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Advanced product recommendation endpoint"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        query = data.get('query', '').strip()
        budget = data.get('budget')
        
        if not user_id or not query:
            return jsonify({"error": "user_id and query are required"}), 400
        
        # Validate budget
        if budget is not None:
            try:
                budget = int(budget)
                if budget <= 0:
                    return jsonify({"error": "Budget must be a positive number"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "Budget must be a valid number"}), 400
        
        # Get recommendations
        result = walmart_ai.recommend_products(user_id, query, budget)
        
        print(f"🎯 Final Result: {len(result.get('product_ids', []))} products recommended")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Server Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "features": ["Enhanced RAG", "Memory", "Agents", "Vector DB", "Smart Matching"],
        "total_products": len(walmart_ai.products)
    })

@app.route('/user-history/<user_id>', methods=['GET'])
def get_user_history(user_id):
    """Get user's shopping history"""
    try:
        history = walmart_ai.get_user_memory(user_id)
        return jsonify({
            "user_id": user_id,
            "total_interactions": len(history),
            "history": history[-5:]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Enhanced Walmart AI Recommendation System Starting...")
    print("✅ Features: Smart Product Matching, RAG, Memory, Agents")
    print("🧠 Enhanced: Better product filtering and AI reasoning")
    print("🎯 Dynamic: Product IDs are now dynamically selected")
    print("📊 Endpoints:")
    print("   - POST /recommend - Get dynamic product recommendations")
    print("   - GET /health - Health check")
    print("   - GET /user-history/<user_id> - Get user history")
    
    app.run(debug=True, host='0.0.0.0', port=5000)