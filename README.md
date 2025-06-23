# Walmart AI Recommendation System

An intelligent product recommendation system powered by Google's Gemini AI that provides personalized shopping recommendations based on user queries, budget constraints, and shopping history.

## 🚀 Features

- **AI-Powered Recommendations**: Uses Google's Gemini 1.5 Flash model for intelligent product matching
- **Budget-Aware Filtering**: Automatically extracts and respects budget constraints from user queries
- **Personalized Experience**: Maintains user shopping history for personalized recommendations
- **Smart Product Matching**: Advanced scoring algorithm that considers product relevance, ratings, and offers
- **Real-time API**: RESTful API endpoints for seamless integration
- **CORS Enabled**: Ready for frontend integration

## 📋 Prerequisites

- Python 3.8+
- Google AI API Key (for Gemini access)
- Flask and required dependencies

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd walmart-ai-recommendation
   ```

2. **Install dependencies**
   ```bash
   pip install flask flask-cors python-dotenv langchain-google-genai langchain
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5000`

## 🎯 API Endpoints

### POST /recommend
Get personalized product recommendations based on user query.

**Request Body:**
```json
{
    "id": "user123",
    "query": "iPhone under 100000"
}
```

**Response:**
```json
{
    "product_ids": [1001, 1015],
    "products": [
        {
            "id": 1001,
            "name": "iPhone 15 Pro",
            "features": "6.1-inch display, A17 Pro chip, 128GB, Pro camera system",
            "price": 99900,
            "offer": "Save $200 + Free shipping",
            "category": "Electronics",
            "brand": "Apple",
            "rating": 4.8,
            "in_stock": true
        }
    ],
    "summary": "Found products matching your query within budget",
    "personalized_note": "Based on your previous searches for electronics"
}
```

### GET /health
Health check endpoint to verify system status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-06-23T10:30:00",
    "total_products": 10
}
```

## 💡 Usage Examples

### Basic Product Search
```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "id": "user123",
    "query": "gaming laptop"
  }'
```

### Budget-Constrained Search
```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "id": "user123",
    "query": "smartphone under 50000"
  }'
```

### Category-Based Search
```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "id": "user123",
    "query": "vitamin c face wash"
  }'
```

## 🧠 How It Works

1. **Query Processing**: Extracts budget constraints and cleans the search query
2. **Product Matching**: Uses advanced scoring algorithm to find relevant products
3. **AI Recommendation**: Leverages Gemini AI to select the best matching products
4. **Personalization**: Considers user's shopping history for tailored recommendations
5. **Response Formatting**: Returns formatted product data with recommendations

## 🏪 Product Categories

The system includes products across multiple categories:
- **Electronics**: Smartphones, laptops, TVs, gaming consoles
- **Home & Kitchen**: Appliances, cookware, cleaning tools
- **Fashion**: Shoes, clothing, accessories
- **Beauty & Personal Care**: Skincare, cosmetics, health products

## 🔧 Configuration

### Adjusting Product Scoring
The system uses a sophisticated scoring algorithm that considers:
- Exact phrase matches in product name/brand (20 points)
- Individual keyword matches in name/brand (8 points)
- Keyword matches in full product text (3 points)
- Product ratings (0.5 multiplier)
- Category relevance (2 points)

### Budget Pattern Recognition
Supports various budget formats:
- "under ₹50000"
- "budget of 30000"
- "within Rs. 25000"
- "less than 40000"
- "max ₹60000"

## 📊 User Memory System

The system maintains user shopping history:
- Stores last 10 interactions per user
- Tracks queries and recommended products
- Provides personalized recommendations based on history
- Automatically expires old data

## 🛡 Error Handling

- Graceful fallback when AI service is unavailable
- Input validation for required fields
- Comprehensive error messages
- Automatic retry mechanisms

## 🔒 Security Features

- Input sanitization
- CORS protection
- Environment variable protection for API keys
- Request validation

## 📈 Performance Optimization

- Efficient product filtering
- Cached user memories
- Optimized scoring algorithm
- Limited result sets to prevent overload

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the API documentation

## 📋 Todo

- [ ] Add product reviews integration
- [ ] Implement collaborative filtering
- [ ] Add inventory management
- [ ] Create admin dashboard
- [ ] Add analytics and reporting
- [ ] Implement caching layer
- [ ] Add rate limiting
- [ ] Create comprehensive test suite

---

**Made by Rahul..**
