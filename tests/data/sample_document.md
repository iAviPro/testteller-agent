# E-commerce Platform API Documentation

## Overview
This document describes the REST API for the E-commerce Platform, including user management, product catalog, shopping cart, and order processing.

## Authentication
All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:
```
Authorization: Bearer <jwt_token>
```

## User Management API

### Register User
- **Endpoint:** `POST /api/auth/register`
- **Description:** Register a new user account
- **Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "first_name": "string",
  "last_name": "string"
}
```
- **Response:** User object with ID and authentication token

### Login User
- **Endpoint:** `POST /api/auth/login`
- **Description:** Authenticate user and return JWT token
- **Request Body:**
```json
{
  "email": "string",
  "password": "string"
}
```
- **Response:** JWT token and user profile

### Get User Profile
- **Endpoint:** `GET /api/users/profile`
- **Description:** Get current user's profile information
- **Authentication:** Required
- **Response:** User profile object

## Product Catalog API

### List Products
- **Endpoint:** `GET /api/products`
- **Description:** Get paginated list of products
- **Query Parameters:**
  - `page`: Page number (default: 1)
  - `limit`: Items per page (default: 20)
  - `category`: Filter by category
  - `search`: Search query
- **Response:** Paginated list of products

### Get Product Details
- **Endpoint:** `GET /api/products/{product_id}`
- **Description:** Get detailed information about a specific product
- **Response:** Product object with full details

### Create Product
- **Endpoint:** `POST /api/products`
- **Description:** Create a new product (Admin only)
- **Authentication:** Required (Admin role)
- **Request Body:**
```json
{
  "name": "string",
  "description": "string",
  "price": "number",
  "category": "string",
  "stock_quantity": "number",
  "images": ["string"]
}
```

## Shopping Cart API

### Get Cart
- **Endpoint:** `GET /api/cart`
- **Description:** Get current user's shopping cart
- **Authentication:** Required
- **Response:** Cart object with items

### Add Item to Cart
- **Endpoint:** `POST /api/cart/items`
- **Description:** Add product to shopping cart
- **Authentication:** Required
- **Request Body:**
```json
{
  "product_id": "string",
  "quantity": "number"
}
```

### Update Cart Item
- **Endpoint:** `PUT /api/cart/items/{item_id}`
- **Description:** Update quantity of cart item
- **Authentication:** Required
- **Request Body:**
```json
{
  "quantity": "number"
}
```

### Remove Cart Item
- **Endpoint:** `DELETE /api/cart/items/{item_id}`
- **Description:** Remove item from cart
- **Authentication:** Required

## Order Processing API

### Create Order
- **Endpoint:** `POST /api/orders`
- **Description:** Create order from current cart
- **Authentication:** Required
- **Request Body:**
```json
{
  "shipping_address": {
    "street": "string",
    "city": "string",
    "state": "string",
    "zip_code": "string",
    "country": "string"
  },
  "payment_method": "string"
}
```

### Get Order Details
- **Endpoint:** `GET /api/orders/{order_id}`
- **Description:** Get order details
- **Authentication:** Required
- **Response:** Order object with items and status

### List User Orders
- **Endpoint:** `GET /api/orders`
- **Description:** Get user's order history
- **Authentication:** Required
- **Response:** Paginated list of orders

## Error Handling
All endpoints return appropriate HTTP status codes and error messages:

- **400 Bad Request:** Invalid request data
- **401 Unauthorized:** Missing or invalid authentication
- **403 Forbidden:** Insufficient permissions
- **404 Not Found:** Resource not found
- **500 Internal Server Error:** Server error

Error response format:
```json
{
  "error": "string",
  "message": "string",
  "details": "object"
}
```

## Rate Limiting
API endpoints are rate limited to prevent abuse:
- **Authenticated users:** 1000 requests per hour
- **Unauthenticated users:** 100 requests per hour

## Security Requirements
- All passwords must be hashed using bcrypt
- JWT tokens expire after 24 hours
- HTTPS required for all endpoints
- Input validation on all endpoints
- SQL injection prevention
- XSS protection 