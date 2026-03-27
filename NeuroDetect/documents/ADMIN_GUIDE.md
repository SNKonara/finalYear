# Admin Management Page - NeuroDetect

## Overview
The Admin Management page provides complete user lifecycle management and monitoring capabilities for administrators. Admins can create, delete, and manage user roles while maintaining access to dashboards and reports (but not fraud investigation or model configuration).

## Location & Access
- **Route**: `/user-management`
- **Default Admin Route**: Admins are automatically routed to `/user-management` upon login
- **Permission**: Requires `admin` role

## Features

### 1. User Management Dashboard

#### Quick Statistics
- **Total Users**: Count of all users in the system
- **Administrators**: Count of admin accounts
- **Analysts**: Count of fraud analyst accounts
- **Viewers**: Count of viewer/reporting-only accounts

#### User Table
Interactive table displaying all users with:
- **Name**: User's full name
- **Email**: User's email address (unique identifier)
- **Role**: Current role badge (color-coded)
  - Purple: Administrator
  - Green: Fraud Analyst
  - Cyan: Viewer

#### Actions per User
- **Edit Role**: Click the pencil icon to change user role
  - Select new role from dropdown
  - Click "Save" to confirm or "Cancel" to discard
- **Delete User**: Click the trash icon to delete
  - Confirmation dialog prevents accidental deletion
  - Cannot delete your own admin account for security

### 2. Add New User

**Access**: Click "Add User" button (top right)

**Form Fields**:
- **Name**: Full name (required)
- **Email**: Unique email address (required, must be valid)
- **Password**: Temporary password (minimum 6 characters)
- **Role**: Initial role assignment
  - Administrator: Full system access
  - Fraud Analyst: Can investigate fraud, access models, stream data
  - Viewer: Can only view reports and dashboards

**Validation**:
- Email must be unique across the system
- Password must be at least 6 characters
- All fields are required
- Real-time error messages on validation failure

### 3. Navigation & Access

**Top Navigation Links**:
- **Dashboard**: Access to unified model dashboard and streaming data
- **Reports**: View compliance, fraud summary, and KPI reports
- **Users**: Current page - User management

**Sidebar Navigation** (Available from navbar):
- Dashboard (Model detection)
- Streaming (Real-time data)
- Batch Upload (Batch predictions)
- Alerts (SNN alerts - restricted)
- Reports (View only)
- Audit (Audit logs)
- Users (This page)

## Permissions Matrix

### Admin Role Capabilities
✓ **CAN DO**:
- Add new users
- Delete existing users
- Update user roles
- View user management interface
- Access dashboards and analytics
- View compliance and fraud reports
- View audit logs
- Access real-time streaming data

✗ **CANNOT DO**:
- Change model configurations
- Investigate fraud cases
- Modify detection thresholds
- Access model training/retraining
- Run batch predictions with custom settings
- Change system settings
- Access fraud investigation tools

### Other Role Access to Admin Page
- **Analyst**: Redirected to unauthorized page
- **Viewer**: Redirected to unauthorized page

## Default User Accounts

When the system initializes, three default users are created:

| Email | Password | Role |
|-------|----------|------|
| admin@neurodetect.ai | admin123 | Administrator |
| analyst@neurodetect.ai | analyst123 | Fraud Analyst |
| viewer@neurodetect.ai | viewer123 | Viewer |

**Note**: Change these default passwords immediately in production.

## Workflow Examples

### Creating a New Analyst
1. Click "Add User" button
2. Enter:
   - Name: "Jane Smith"
   - Email: "jane.smith@company.com"
   - Password: "SecurePass123"
   - Role: "Fraud Analyst"
3. Click "Create User"
4. Success message confirms user creation

### Updating User Role
1. Find user in table
2. Click pencil icon
3. Select new role from dropdown
4. Click "Save"
5. User's access level updates immediately

### Deleting a User
1. Click trash icon for user
2. Confirmation dialog appears
3. Click "Delete" to confirm
4. User account is removed
5. User's active sessions are invalidated

## Status Messages

### Success Messages
- ✓ "User {email} created successfully"
- ✓ "User role updated successfully"
- ✓ "User deleted successfully"

### Error Messages
- ⚠ "Invalid email or password"
- ⚠ "Session expired. Please login again."
- ⚠ "A user with this email already exists"
- ⚠ "Admin users cannot delete their own account"
- ⚠ "User not found"

## Backend API Reference

All operations require `Bearer {token}` authentication header.

### GET /auth/users
**Permission**: Admin only
**Response**:
```json
{
  "users": [
    {
      "id": "user_id",
      "name": "User Name",
      "email": "user@example.com",
      "role": "admin|analyst|viewer"
    }
  ]
}
```

### POST /auth/users
**Permission**: Admin only
**Request**:
```json
{
  "name": "New User",
  "email": "new@example.com",
  "password": "password123",
  "role": "viewer"
}
```

### PATCH /auth/users/{user_id}/role
**Permission**: Admin only
**Request**:
```json
{
  "role": "analyst"
}
```

### DELETE /auth/users/{user_id}
**Permission**: Admin only
**Note**: Admins cannot delete their own account

## UI Layout

```
┌─────────────────────────────────────────────────┐
│  NeuroDetect Admin Dashboard                    │  [Logout]
│  User & Account Management                      │
└─────────────────────────────────────────────────┘

Quick Stats Row:
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Tot: 10  │ │ Adm: 2   │ │ Ana: 5   │ │ Vie: 3   │
└──────────┘ └──────────┘ └──────────┘ └──────────┘

Controls:
[+ Add User] [Reports] [Dashboard]

User Table:
┌─────────────────────────────────────────────────┐
│ Name    │ Email           │ Role    │ Actions   │
├─────────────────────────────────────────────────┤
│ User 1  │ user1@ex.com   │ Admin   │ ✏️  🗑️    │
│ User 2  │ user2@ex.com   │ Analyst │ ✏️  🗑️    │
└─────────────────────────────────────────────────┘

Admin Permissions Section:
✓ You Can                   ✗ You Cannot
- Add users                 - Change model config
- Delete users              - Investigate fraud
- Assign roles              - Modify thresholds
```

## Security & Best Practices

1. **Session Management**: Admin sessions timeout after inactivity
2. **Password Policy**: Minimum 6 characters (configure as needed)
3. **Account Security**: Admins cannot delete their own account
4. **Audit Trail**: All user management actions are logged
5. **Role Separation**: Each role has distinct capabilities

## Troubleshooting

### "Session expired" message
- Your authentication token has expired
- Click logout and log back in

### "User with this email already exists"
- The email is already assigned to another user
- Use a different email address

### Cannot delete user
- You may be attempting to delete your own admin account
- Use another admin account to delete yours
- Or ask another administrator for help

### Role change not reflecting immediately
- Clear browser cache or refresh the page
- The change takes effect on the user's next action

## Related Documentation
- **Authentication**: See Login page documentation
- **Fraud Investigation**: See Analyst role documentation
- **Reporting**: See Reports page documentation
- **Audit**: See Audit Details page documentation
