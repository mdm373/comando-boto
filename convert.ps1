
Get-Content .env | foreach {
  $name, $value = $_.split('=')
  if ([string]::IsNullOrWhiteSpace($name) -or $name.Contains('#')) {
    # skip empty or comment line in ENV file
    return
  }
  Set-Content env:\$name $value
}

$out_file="$Env:GGUF_DIR\commando-boto.gguf"
uv run --directory ../llama.cpp --no-project convert_hf_to_gguf.py ..\comando-boto\.dist\ --outfile "$out_file"