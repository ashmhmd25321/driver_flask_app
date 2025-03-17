#!/bin/bash
# Script to set up Let's Encrypt SSL certificates for the Flask API

# Check if domain name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 yourdomain.com [email@example.com]"
    exit 1
fi

DOMAIN=$1
EMAIL=$2

# Install certbot if not already installed
if ! command -v certbot &> /dev/null; then
    echo "Installing certbot..."
    sudo apt-get update
    sudo apt-get install -y certbot
fi

# Create ssl directory if it doesn't exist
mkdir -p ssl

# Get certificate from Let's Encrypt
if [ -z "$EMAIL" ]; then
    sudo certbot certonly --standalone --preferred-challenges http -d $DOMAIN --agree-tos --register-unsafely-without-email
else
    sudo certbot certonly --standalone --preferred-challenges http -d $DOMAIN --agree-tos --email $EMAIL
fi

# Check if certificate was obtained successfully
if [ -d "/etc/letsencrypt/live/$DOMAIN" ]; then
    # Copy certificates to the application's ssl directory
    sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem ssl/cert.pem
    sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem ssl/key.pem
    
    # Set proper permissions
    sudo chmod 644 ssl/cert.pem
    sudo chmod 600 ssl/key.pem
    
    echo "SSL certificates have been set up successfully!"
    echo "You can now run your Flask application with HTTPS support."
    echo "Don't forget to open port 443 in your firewall/security group."
else
    echo "Failed to obtain SSL certificates. Please check the error messages above."
fi

# Add a cron job for automatic renewal
echo "Setting up automatic renewal..."
(crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet && cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $(pwd)/ssl/cert.pem && cp /etc/letsencrypt/live/$DOMAIN/privkey.pem $(pwd)/ssl/key.pem") | crontab -

echo "Setup complete! Your certificates will be automatically renewed." 