using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using UdcFileEncryption.Models;

namespace UdcFileEncryption.Services;

/// <summary>
/// ON-PREM encryption service.
/// Uses RSA public key (from Config JSON) + AES-256-GCM envelope encryption.
/// Cannot decrypt — only the cloud side with the private key can.
/// </summary>
public class OnPremEncryptionService
{
    private readonly RSA _publicKey;
    private readonly UdcEncryptionConfig _config;

    public OnPremEncryptionService(UdcEncryptionConfig config)
    {
        _config = config;

        // Load the RSA public key from PEM file
        _publicKey = RSA.Create();
        var pemContent = File.ReadAllText(config.PublicKeyPemPath);
        _publicKey.ImportFromPem(pemContent);

        Console.WriteLine($"[OnPrem] Loaded RSA public key: {_publicKey.KeySize}-bit");
        Console.WriteLine($"[OnPrem] Site ID: {config.SiteId}");
        Console.WriteLine($"[OnPrem] Key Version: {config.KeyVersion}");
    }

    /// <summary>
    /// Encrypts a file using envelope encryption.
    /// 
    /// File layout:
    ///   [Header JSON + \n]           — plaintext, authenticated via AAD
    ///   [4 bytes envelope length]    — little-endian uint32
    ///   [RSA-OAEP encrypted envelope] — contains AES key (32) + Nonce (12)
    ///   [16 bytes GCM auth tag]      — integrity proof
    ///   [AES-256-GCM ciphertext]     — encrypted file content
    /// </summary>
    public void EncryptFile(string inputFilePath, string outputFilePath, string fileCategory)
    {
        // Read the original file content
        byte[] plaintext = File.ReadAllBytes(inputFilePath);
        Console.WriteLine($"[OnPrem] Input file: {inputFilePath} ({plaintext.Length:N0} bytes)");

        // === Step 1: Build the plaintext header ===
        var header = new EncryptedFileHeader
        {
            Version = _config.KeyVersion,
            SiteId = _config.SiteId,
            Timestamp = DateTime.UtcNow.ToString("yyyyMMddTHHmmssZ"),
            Category = fileCategory
        };
        byte[] headerBytes = Encoding.UTF8.GetBytes(
            JsonSerializer.Serialize(header, new JsonSerializerOptions { WriteIndented = false })
        );

        // === Step 2: Generate random AES-256 key and nonce ===
        byte[] aesKey = RandomNumberGenerator.GetBytes(32);   // 256-bit AES key
        byte[] nonce = RandomNumberGenerator.GetBytes(12);     // 96-bit GCM nonce

        // === Step 3: Encrypt file content with AES-256-GCM ===
        //             Header is used as AAD (authenticated but not encrypted)
        byte[] ciphertext = new byte[plaintext.Length];
        byte[] authTag = new byte[16];  // 128-bit GCM authentication tag

        using (var aesGcm = new AesGcm(aesKey, 16))
        {
            aesGcm.Encrypt(
                nonce: nonce,
                plaintext: plaintext,
                ciphertext: ciphertext,
                tag: authTag,
                associatedData: headerBytes  // AAD — tamper-proofs the header
            );
        }

        // === Step 4: Encrypt the AES key + nonce with RSA public key ===
        //             This is the "envelope" — only Key Vault can unwrap it
        byte[] keyMaterial = new byte[32 + 12];  // AES key + nonce
        Buffer.BlockCopy(aesKey, 0, keyMaterial, 0, 32);
        Buffer.BlockCopy(nonce, 0, keyMaterial, 32, 12);

        byte[] encryptedEnvelope = _publicKey.Encrypt(keyMaterial, RSAEncryptionPadding.OaepSHA256);

        // === Step 5: Write the encrypted file ===
        using var output = File.Create(outputFilePath);

        // Section 1: Header JSON + newline (plaintext)
        output.Write(headerBytes);
        output.WriteByte((byte)'\n');

        // Section 2: Envelope length (4 bytes, little-endian) + encrypted envelope
        output.Write(BitConverter.GetBytes((uint)encryptedEnvelope.Length));
        output.Write(encryptedEnvelope);

        // Section 3: GCM authentication tag (16 bytes)
        output.Write(authTag);

        // Section 4: AES-GCM ciphertext
        output.Write(ciphertext);

        // Clear sensitive key material from memory
        CryptographicOperations.ZeroMemory(aesKey);
        CryptographicOperations.ZeroMemory(keyMaterial);

        var fileInfo = new FileInfo(outputFilePath);
        Console.WriteLine($"[OnPrem] Encrypted file: {outputFilePath} ({fileInfo.Length:N0} bytes)");
        Console.WriteLine($"[OnPrem] Overhead: {fileInfo.Length - plaintext.Length} bytes " +
                          $"(header:{headerBytes.Length + 1} + envelope:{encryptedEnvelope.Length + 4} + tag:16)");
    }
}
