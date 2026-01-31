FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json ./

# Install with legacy peer deps
RUN npm install --legacy-peer-deps

# Copy app files
COPY . .

# Expose port
EXPOSE 3000

# Start
CMD ["npm", "start"]