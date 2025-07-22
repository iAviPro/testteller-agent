# Sample Test Cases

## Test Case 1: User Authentication
**Test ID**: TC001  
**Priority**: High  
**Category**: Authentication  

### Test Description
Verify that a user can successfully log in with valid credentials.

### Pre-conditions
- User account exists in the system
- Login page is accessible

### Test Steps
1. Navigate to the login page
2. Enter valid username: `testuser@example.com`
3. Enter valid password: `ValidPass123!`
4. Click the "Login" button

### Expected Results
- User is successfully authenticated
- User is redirected to the dashboard
- Welcome message displays user's name
- Session token is created

### Test Data
```json
{
  "username": "testuser@example.com",
  "password": "ValidPass123!",
  "expected_redirect": "/dashboard"
}
```

---

## Test Case 2: Invalid Login Attempt
**Test ID**: TC002  
**Priority**: Medium  
**Category**: Authentication  

### Test Description
Verify that login fails with invalid credentials and appropriate error message is shown.

### Pre-conditions
- Login page is accessible

### Test Steps
1. Navigate to the login page
2. Enter invalid username: `invalid@example.com`
3. Enter invalid password: `wrongpassword`
4. Click the "Login" button

### Expected Results
- Login attempt fails
- Error message: "Invalid username or password"
- User remains on login page
- No session token is created

### Test Data
```json
{
  "username": "invalid@example.com",
  "password": "wrongpassword",
  "expected_error": "Invalid username or password"
}
```

---

## Test Case 3: Product Search Functionality
**Test ID**: TC003  
**Priority**: Medium  
**Category**: Search  

### Test Description
Verify that users can search for products and receive relevant results.

### Pre-conditions
- User is logged in
- Products exist in the database
- Search functionality is available

### Test Steps
1. Navigate to the products page
2. Enter search term: `laptop`
3. Click the "Search" button or press Enter
4. Review the search results

### Expected Results
- Search results are displayed
- Results contain products matching "laptop"
- Results are sorted by relevance
- Total number of results is shown

### Test Data
```json
{
  "search_term": "laptop",
  "expected_min_results": 5,
  "expected_categories": ["Electronics", "Computers"]
}
```

---

## Test Case 4: Add Item to Shopping Cart
**Test ID**: TC004  
**Priority**: High  
**Category**: E-commerce  

### Test Description
Verify that users can successfully add items to their shopping cart.

### Pre-conditions
- User is logged in
- Product is available in inventory
- Shopping cart is accessible

### Test Steps
1. Navigate to a product detail page
2. Select product quantity: `2`
3. Click "Add to Cart" button
4. View shopping cart

### Expected Results
- Product is added to cart with correct quantity
- Cart total is updated
- Success message is displayed
- Cart icon shows updated item count

### Test Data
```json
{
  "product_id": "LAPTOP001",
  "product_name": "Gaming Laptop Pro",
  "quantity": 2,
  "unit_price": 1299.99,
  "expected_total": 2599.98
}
```