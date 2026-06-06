using System.Text.Json.Serialization;

namespace UdcFileEncryption.Models;

/// <summary>
/// Plaintext header embedded in every encrypted UDC file.
/// Readable without decryption (for routing/validation).
/// Tamper-proof via AES-GCM AAD authentication.
/// </summary>
public class EncryptedFileHeader
{
    /// <summary>Key version — maps to Key Vault key name. E.g., "UDC_KEY_V_1.0"</summary>
    [JsonPropertyName("v")]
    public string Version { get; set; } = string.Empty;

    /// <summary>Site identifier — for wrong-site upload prevention</summary>
    [JsonPropertyName("s")]
    public string SiteId { get; set; } = string.Empty;

    /// <summary>UTC timestamp when the file was encrypted</summary>
    [JsonPropertyName("ts")]
    public string Timestamp { get; set; } = string.Empty;

    /// <summary>File category (e.g., "ES_SSHLOG", "ES_EXPERIONEVENTLOGS")</summary>
    [JsonPropertyName("cat")]
    public string Category { get; set; } = string.Empty;
}
