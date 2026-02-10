# Copyright 2025 go-highway Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile for building and testing go-highway
#
# Build: docker build -t go-highway .
# Test:  docker run --rm go-highway

FROM golang:1.26 AS builder

ENV GOEXPERIMENT=simd

WORKDIR /app

# Copy go.mod and go.sum first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build hwygen
RUN go build -o bin/hwygen ./cmd/hwygen

# Run go generate on examples
RUN PATH="/app/bin:$PATH" go generate ./examples/...

# Build all packages
RUN go build ./...

# Run tests
FROM builder AS tester

# Run all tests
RUN go test ./... -v

# Run tests with fallback (HWY_NO_SIMD)
RUN HWY_NO_SIMD=1 go test ./... -v

# Final stage - just verify build succeeded
FROM builder AS final

# Default command runs tests
CMD ["sh", "-c", "go test ./..."]
