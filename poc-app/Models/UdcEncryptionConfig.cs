namespace UdcFileEncryption.Models;

/// <summary>
/// Configuration loaded from Config JSON (downloaded from dashboard).
/// Contains Site ID + encryption public key + key version.
/// </summary>
public class UdcEncryptionConfig
{
    public string SiteId { get; set; } = string.Empty;
    public string KeyVersion { get; set; } = string.Empty;
    public string PublicKeyPemPath { get; set; } = string.Empty;
    public DateTime KeyExpiresAt { get; set; }
}
