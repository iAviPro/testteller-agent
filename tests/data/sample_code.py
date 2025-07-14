"""
E-commerce Platform - User Authentication Service

This module provides user authentication functionality including
registration, login, password reset, and JWT token management.
"""

import hashlib
import jwt
import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class UserRole(Enum):
    """User role enumeration."""
    CUSTOMER = "customer"
    ADMIN = "admin"
    MODERATOR = "moderator"


@dataclass
class User:
    """User data model."""
    id: str
    username: str
    email: str
    password_hash: str
    first_name: str
    last_name: str
    role: UserRole
    created_at: datetime.datetime
    last_login: Optional[datetime.datetime] = None
    is_active: bool = True
    email_verified: bool = False


class AuthenticationError(Exception):
    """Authentication related errors."""
    pass


class ValidationError(Exception):
    """Data validation errors."""
    pass


class UserAuthService:
    """User authentication service."""

    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        """
        Initialize authentication service.

        Args:
            secret_key: Secret key for JWT token signing
            token_expiry_hours: JWT token expiry time in hours
        """
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        # In-memory user storage (use proper DB in production)
        self.users_db = {}

    def hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify password against hash.

        Args:
            password: Plain text password
            password_hash: Stored password hash

        Returns:
            True if password matches, False otherwise
        """
        return self.hash_password(password) == password_hash

    def validate_email(self, email: str) -> bool:
        """
        Validate email format.

        Args:
            email: Email address to validate

        Returns:
            True if email is valid, False otherwise
        """
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def validate_password(self, password: str) -> bool:
        """
        Validate password strength.

        Args:
            password: Password to validate

        Returns:
            True if password meets requirements, False otherwise
        """
        if len(password) < 8:
            return False

        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        return has_upper and has_lower and has_digit and has_special

    def register_user(self, username: str, email: str, password: str,
                      first_name: str, last_name: str, role: UserRole = UserRole.CUSTOMER) -> User:
        """
        Register a new user.

        Args:
            username: Unique username
            email: User email address
            password: Plain text password
            first_name: User's first name
            last_name: User's last name
            role: User role (default: CUSTOMER)

        Returns:
            Created user object

        Raises:
            ValidationError: If validation fails
            AuthenticationError: If user already exists
        """
        # Validate input
        if not self.validate_email(email):
            raise ValidationError("Invalid email format")

        if not self.validate_password(password):
            raise ValidationError("Password does not meet requirements")

        if username in self.users_db:
            raise AuthenticationError("Username already exists")

        # Check if email already exists
        for user in self.users_db.values():
            if user.email == email:
                raise AuthenticationError("Email already registered")

        # Create user
        user = User(
            id=f"user_{len(self.users_db) + 1}",
            username=username,
            email=email,
            password_hash=self.hash_password(password),
            first_name=first_name,
            last_name=last_name,
            role=role,
            created_at=datetime.datetime.now()
        )

        self.users_db[username] = user
        return user

    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user and return JWT token.

        Args:
            username: Username or email
            password: Plain text password

        Returns:
            Dictionary containing JWT token and user info

        Raises:
            AuthenticationError: If authentication fails
        """
        # Find user by username or email
        user = None
        if username in self.users_db:
            user = self.users_db[username]
        else:
            # Search by email
            for u in self.users_db.values():
                if u.email == username:
                    user = u
                    break

        if not user:
            raise AuthenticationError("User not found")

        if not user.is_active:
            raise AuthenticationError("User account is disabled")

        if not self.verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid password")

        # Update last login
        user.last_login = datetime.datetime.now()

        # Generate JWT token
        token = self.generate_jwt_token(user)

        return {
            "token": token,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": user.role.value
            }
        }

    def generate_jwt_token(self, user: User) -> str:
        """
        Generate JWT token for user.

        Args:
            user: User object

        Returns:
            JWT token string
        """
        payload = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=self.token_expiry_hours)
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User object if found, None otherwise
        """
        for user in self.users_db.values():
            if user.id == user_id:
                return user
        return None

    def update_user_profile(self, user_id: str, **kwargs) -> User:
        """
        Update user profile information.

        Args:
            user_id: User ID
            **kwargs: Fields to update

        Returns:
            Updated user object

        Raises:
            AuthenticationError: If user not found
        """
        user = self.get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found")

        # Update allowed fields
        allowed_fields = ["first_name", "last_name", "email"]
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(user, field, value)

        return user

    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password

        Returns:
            True if password changed successfully

        Raises:
            AuthenticationError: If user not found or old password is incorrect
            ValidationError: If new password is invalid
        """
        user = self.get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found")

        if not self.verify_password(old_password, user.password_hash):
            raise AuthenticationError("Current password is incorrect")

        if not self.validate_password(new_password):
            raise ValidationError("New password does not meet requirements")

        user.password_hash = self.hash_password(new_password)
        return True

    def deactivate_user(self, user_id: str) -> bool:
        """
        Deactivate user account.

        Args:
            user_id: User ID

        Returns:
            True if user deactivated successfully

        Raises:
            AuthenticationError: If user not found
        """
        user = self.get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found")

        user.is_active = False
        return True

    def get_user_count(self) -> int:
        """
        Get total number of registered users.

        Returns:
            Number of users
        """
        return len(self.users_db)

    def get_active_users(self) -> list[User]:
        """
        Get list of active users.

        Returns:
            List of active user objects
        """
        return [user for user in self.users_db.values() if user.is_active]
