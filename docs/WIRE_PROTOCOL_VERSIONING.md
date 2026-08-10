# Wire Protocol Versioning Policy

This document defines the versioning strategy for Vulkan-VM's network wire protocol.

## Version Format

Protocol versions follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (incompatible wire format)
- **MINOR**: Backward-compatible feature additions
- **PATCH**: Bug fixes, no wire format changes

## Version Negotiation

1. **ClientHello** / **ServerHello** messages contain `protocol_version` field (uint32_t packed as `MAJOR<<16 | MINOR<<8 | PATCH`)
2. Server responds with highest mutually supported version
3. If no compatible version, connection is rejected with `PROTOCOL_MISMATCH` error

## Current Version

- **Version**: 1.0.0 (0x00010000)
- **Status**: Stable

## Compatibility Rules

### MAJOR version bump required for:
- Changing message field order, types, or sizes
- Removing message types or fields
- Changing enum values that cross node boundaries
- Changing serialization format (e.g., endianness, alignment)

### MINOR version bump for:
- Adding new optional message types
- Adding new fields to existing messages (with defaults)
- New capabilities announced in handshake
- New RDMA transport features

### PATCH version bump for:
- Bug fixes in protocol implementation
- Documentation updates
- Performance improvements without wire changes

## Message Versioning

Each message type has an independent version field. Receivers must:
- Accept messages with `message_version <= receiver_max_version`
- Ignore unknown fields in known message types
- Reject messages with `message_version > receiver_max_version`

## Deprecation Policy

- Deprecated message types/fields marked in spec with `DEPRECATED` tag
- Minimum 2 MAJOR versions before removal
- Deprecation announced in release notes

## Implementation

See `include/vulkan_vm/network/wire_protocol.hpp` for protocol definitions.

Version constants defined in `include/vulkan_vm/constants.hpp`:
- `VVM_PROTOCOL_VERSION_MAJOR`
- `VVM_PROTOCOL_VERSION_MINOR`
- `VVM_PROTOCOL_VERSION_PATCH`
- `VVM_PROTOCOL_VERSION` (packed)

## Testing

- CI runs protocol compatibility tests between adjacent versions
- New versions must pass `network_test` with previous version's test vectors