{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python e pip
    python311
    python311Packages.pip

    # OpenCV com suporte para webcam e GUI
    python311Packages.opencv4

    # Dependências principais do script
    python311Packages.numpy
    python311Packages.tensorflow

    # Dependências do sistema para OpenCV/webcam
    libv4l
    v4l-utils

    # Bibliotecas gráficas necessárias para cv2.imshow
    xorg.libX11
    xorg.libXext
    gtk3
    glib

    # Dependências opcionais que podem ser úteis
    python311Packages.pillow
    python311Packages.matplotlib
  ];

  # Variáveis de ambiente
  shellHook = ''
    echo "🚀 Ambiente Nix para Detector de Atenção"
    echo "======================================"
    echo "Python: $(python --version)"
    echo "OpenCV: $(python -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Não instalado")"
    echo "TensorFlow: $(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "Não instalado")"
    echo ""
    echo "💡 Para instalar dependências extras:"
    echo "pip install --user keras pillow"
    echo ""
    echo "🎥 Para testar webcam:"
    echo "python -c \"import cv2; print('OpenCV OK')\" && echo '✅ OpenCV funcionando'"
    echo ""
    echo "▶️  Para rodar o script:"
    echo "python detector_webcam.py"
    echo "======================================"

    # Configurar PYTHONPATH se necessário
    export PYTHONPATH="$PWD:$PYTHONPATH"

    # Configurar para encontrar bibliotecas do sistema
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.libv4l
      pkgs.xorg.libX11
      pkgs.xorg.libXext
      pkgs.gtk3
      pkgs.glib
    ]}:$LD_LIBRARY_PATH"
  '';

  # Permitir pacotes não-livres se necessário (para CUDA/etc)
  nixpkgs.config.allowUnfree = true;
}
